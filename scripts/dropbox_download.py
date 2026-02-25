#!/usr/bin/env python3
"""
dropbox_download.py — Download a Dropbox folder (or shared link) to local disk.

Features:
  - Download from a shared link URL or a Dropbox path
  - Recursive folder download with directory structure preserved
  - Skip files that already exist locally (--skip-existing)
  - Progress display
  - Chunked download for large files (low memory)
  - Robust retry with exponential backoff

Usage:
  # Download from a shared folder link
  python dropbox_download.py \\
      --shared-link "https://www.dropbox.com/scl/fo/xxx/yyy?rlkey=zzz&dl=0" \\
      --local /path/to/destination

  # Download a specific Dropbox path
  python dropbox_download.py --remote /datasets/goal1 --local ./goal1

  # Resume a partial download
  python dropbox_download.py --shared-link "..." --local ./goal1 --skip-existing
"""

import argparse
import os
import sys
import time
from pathlib import Path

import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import FolderMetadata, FileMetadata

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 5
BACKOFF_BASE = 2
DOWNLOAD_CHUNK = 64 * 1024 * 1024  # 64 MB read chunks for streaming download


# ── Auth helpers ──────────────────────────────────────────────────────────────

def get_token(args):
    """Resolve Dropbox access token from args, env, or file."""
    if args.token:
        return args.token
    if os.environ.get("DROPBOX_ACCESS_TOKEN"):
        return os.environ["DROPBOX_ACCESS_TOKEN"]
    token_file = Path.home() / ".dropbox_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def get_dbx(args):
    """Create an authenticated Dropbox client."""
    token = get_token(args)

    if args.refresh_token and args.app_key:
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=args.refresh_token,
            app_key=args.app_key,
            app_secret=args.app_secret or "",
        )
    elif token:
        dbx = dropbox.Dropbox(token)
    else:
        print("ERROR: No Dropbox token found.")
        print("Provide via --token, DROPBOX_ACCESS_TOKEN env, or ~/.dropbox_token")
        sys.exit(1)

    try:
        acct = dbx.users_get_current_account()
        print(f"Authenticated as: {acct.name.display_name} ({acct.email})")
    except AuthError:
        print("ERROR: Invalid Dropbox token. Please generate a new one.")
        sys.exit(1)

    return dbx


# ── Retry helper ──────────────────────────────────────────────────────────────

def _retry(fn, *a, **kw):
    """Retry a Dropbox API call with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*a, **kw)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            err_str = str(e)
            if "rate_limit" in err_str:
                wait = BACKOFF_BASE ** (attempt + 1)
            else:
                wait = BACKOFF_BASE ** attempt
            print(f"    Retrying in {wait}s: {e}")
            time.sleep(wait)


# ── Listing helpers ───────────────────────────────────────────────────────────

def list_folder_recursive(dbx, path, shared_link=None):
    """List all files in a Dropbox folder recursively.

    Returns list of (dropbox_path, size) tuples for files only.
    """
    files = []
    sl = dropbox.files.SharedLink(url=shared_link) if shared_link else None

    try:
        if shared_link:
            result = _retry(
                dbx.files_list_folder,
                path=path,
                recursive=True,
                shared_link=sl,
            )
        else:
            result = _retry(
                dbx.files_list_folder,
                path=path,
                recursive=True,
            )
    except ApiError as e:
        print(f"ERROR listing folder: {e}")
        return files

    while True:
        for entry in result.entries:
            if isinstance(entry, FileMetadata):
                files.append((entry.path_display, entry.size))
        if not result.has_more:
            break
        result = _retry(dbx.files_list_folder_continue, result.cursor)

    return files


# ── Download helpers ──────────────────────────────────────────────────────────

def download_file(dbx, remote_path, local_path, file_size=0, shared_link=None):
    """Download a single file from Dropbox, streaming to avoid high memory use."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    sl = dropbox.files.SharedLink(url=shared_link) if shared_link else None

    if shared_link:
        meta, response = _retry(
            dbx.sharing_get_shared_link_file,
            url=shared_link,
            path=remote_path,
        )
    else:
        meta, response = _retry(dbx.files_download, remote_path)

    tmp_path = local_path + ".tmp"
    downloaded = 0

    try:
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(DOWNLOAD_CHUNK):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        pct = downloaded / file_size * 100
                        print(f"\r    {downloaded / 1e6:,.0f} / "
                              f"{file_size / 1e6:,.0f} MB ({pct:.1f}%)",
                              end="", flush=True)
        # Atomic rename
        os.replace(tmp_path, local_path)
        if file_size > 0:
            print()  # newline after progress
    except Exception:
        # Clean up partial download
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ── Main download logic ──────────────────────────────────────────────────────

def download_folder(dbx, remote_root, local_root, shared_link=None,
                    skip_existing=False):
    """Download an entire Dropbox folder to local disk."""
    print(f"\nScanning Dropbox folder: {remote_root or '(shared link root)'}")

    # For shared links, use "" as the path prefix to list from root of shared folder
    list_path = "" if shared_link else remote_root
    files = list_folder_recursive(dbx, list_path, shared_link=shared_link)

    if not files:
        print("No files found.")
        return

    total_size = sum(s for _, s in files)
    print(f"Files to download: {len(files)}  ({total_size / 1e9:.2f} GB)\n")

    downloaded_count = 0
    skipped_count = 0
    failed = []
    cumulative_bytes = 0

    for i, (remote_path, file_size) in enumerate(files):
        # Compute local path
        if shared_link:
            # For shared links, remote_path is relative to shared folder root
            rel_path = remote_path.lstrip("/")
        else:
            # Strip remote_root prefix to get relative path
            if remote_path.lower().startswith(remote_root.lower()):
                rel_path = remote_path[len(remote_root):].lstrip("/")
            else:
                rel_path = remote_path.lstrip("/")

        local_path = os.path.join(local_root, rel_path)
        size_mb = file_size / 1e6

        # Skip check
        if skip_existing and os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if local_size == file_size:
                skipped_count += 1
                cumulative_bytes += file_size
                continue

        print(f"  [{i+1}/{len(files)}] {rel_path}  ({size_mb:,.1f} MB)")

        try:
            download_file(
                dbx, remote_path, local_path, file_size,
                shared_link=shared_link,
            )
            downloaded_count += 1
            cumulative_bytes += file_size
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append((rel_path, str(e)))
            cumulative_bytes += file_size  # count toward progress anyway

        # Overall progress
        overall_pct = cumulative_bytes / total_size * 100 if total_size > 0 else 100
        print(f"    Overall: {overall_pct:.1f}% ({cumulative_bytes / 1e9:.2f} / "
              f"{total_size / 1e9:.2f} GB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Skipped:    {skipped_count}")
    print(f"  Failed:     {len(failed)}")
    if failed:
        for f, e in failed:
            print(f"    ✗ {f}: {e}")
    print(f"  Destination: {local_root}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download a Dropbox folder or shared link to local disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --shared-link "https://www.dropbox.com/scl/fo/..." --local ./data
  %(prog)s --remote /datasets/goal1 --local ./goal1
  %(prog)s --shared-link "..." --local ./data --skip-existing
        """,
    )
    parser.add_argument("--local", required=True,
                        help="Local destination folder")
    parser.add_argument("--remote", default=None,
                        help="Dropbox path to download (e.g. /datasets/goal1)")
    parser.add_argument("--shared-link", default=None,
                        help="Dropbox shared folder/file link URL")
    parser.add_argument("--token", default=None,
                        help="Dropbox access token")
    parser.add_argument("--refresh-token", default=None,
                        help="Dropbox refresh token (long-lived)")
    parser.add_argument("--app-key", default=None,
                        help="Dropbox app key (used with refresh token)")
    parser.add_argument("--app-secret", default=None,
                        help="Dropbox app secret (used with refresh token)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already exist locally with matching size")
    args = parser.parse_args()

    if not args.remote and not args.shared_link:
        sys.exit("ERROR: Provide either --remote or --shared-link")

    dbx = get_dbx(args)

    os.makedirs(args.local, exist_ok=True)

    if args.shared_link:
        download_folder(
            dbx,
            remote_root="",
            local_root=args.local,
            shared_link=args.shared_link,
            skip_existing=args.skip_existing,
        )
    else:
        download_folder(
            dbx,
            remote_root=args.remote,
            local_root=args.local,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
