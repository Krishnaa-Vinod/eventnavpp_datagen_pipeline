#!/usr/bin/env python3
"""
dropbox_upload.py — Upload a local folder (or file) to Dropbox.

Features:
  - Chunked upload for large files (>150 MB) — works for multi-GB H5 files
  - Resume interrupted uploads
  - Skip files that already exist (--skip-existing)
  - Progress bars per file and overall
  - Robust retry with exponential backoff
  - Auth via access token, refresh token, or token file

Authentication setup (one-time):
  1. Go to https://www.dropbox.com/developers/apps
  2. Create an app → choose "Scoped access" → "Full Dropbox"
  3. Under Permissions, enable: files.content.write, files.content.read,
     sharing.read, sharing.write
  4. Under Settings, generate an access token (short-lived, 4 hours)
     OR note the App key + App secret and use the refresh-token flow.
  5. Pass via --token, or set DROPBOX_ACCESS_TOKEN env var,
     or save to ~/.dropbox_token

Usage:
  # Upload a folder to Dropbox root
  python dropbox_upload.py --local /path/to/folder --remote /my_folder

  # Upload to a shared folder (auto-resolves shared link → Dropbox path)
  python dropbox_upload.py --local /path/to/folder \\
      --shared-link "https://www.dropbox.com/scl/fo/xxx/yyy?rlkey=zzz&dl=0"

  # With explicit token
  python dropbox_upload.py --local ./data --remote /data --token "sl.xxxxx"

  # Skip existing files (resume a partial upload)
  python dropbox_upload.py --local ./data --remote /data --skip-existing
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import WriteMode, CommitInfo, UploadSessionCursor

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 128 * 1024 * 1024  # 128 MB chunks (Dropbox max session chunk: 150 MB)
SMALL_FILE_LIMIT = 150 * 1024 * 1024  # Files ≤ 150 MB use simple upload
MAX_RETRIES = 5
BACKOFF_BASE = 2  # seconds


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
        # Use refresh token (long-lived) → auto-refreshes access tokens
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

    # Verify auth
    try:
        acct = dbx.users_get_current_account()
        print(f"Authenticated as: {acct.name.display_name} ({acct.email})")
    except AuthError:
        print("ERROR: Invalid Dropbox token. Please generate a new one.")
        sys.exit(1)

    return dbx


# ── Shared link → Dropbox path resolution ─────────────────────────────────────

def resolve_shared_link_path(dbx, shared_link_url):
    """Convert a Dropbox shared link URL to an internal Dropbox path.

    For shared folders, returns the path where the folder is mounted
    in the user's Dropbox.  For shared links to files, returns the
    file's path.
    """
    try:
        meta = dbx.sharing_get_shared_link_metadata(shared_link_url)
        path = meta.path_lower
        if path:
            print(f"Resolved shared link → Dropbox path: {path}")
            return path
        # If path is empty, the shared folder might need to be mounted
        if hasattr(meta, 'name'):
            # Try to find it in the root
            fallback = f"/{meta.name}"
            print(f"Shared link has no direct path; using name: {fallback}")
            return fallback
    except ApiError as e:
        print(f"WARNING: Could not resolve shared link: {e}")
        print("The folder may not be added to your Dropbox account.")
        print("Go to the shared link in a browser and click 'Add to my Dropbox'.")
    return None


# ── Upload helpers ────────────────────────────────────────────────────────────

def _retry(fn, *a, **kw):
    """Retry a Dropbox API call with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*a, **kw)
        except Exception as e:
            err_str = str(e)
            if attempt == MAX_RETRIES - 1:
                raise
            if "too_many_write_operations" in err_str or "rate_limit" in err_str:
                wait = BACKOFF_BASE ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            elif "conflict" in err_str:
                raise  # Don't retry conflicts
            else:
                wait = BACKOFF_BASE ** attempt
                print(f"    Retrying in {wait}s: {e}")
                time.sleep(wait)


def upload_small_file(dbx, local_path, remote_path, write_mode):
    """Upload a file ≤ 150 MB in a single request."""
    with open(local_path, "rb") as f:
        data = f.read()
    _retry(dbx.files_upload, data, remote_path, mode=write_mode)


def upload_large_file(dbx, local_path, remote_path, write_mode):
    """Upload a file > 150 MB using chunked upload sessions."""
    file_size = os.path.getsize(local_path)
    uploaded = 0

    with open(local_path, "rb") as f:
        # Start session
        chunk = f.read(CHUNK_SIZE)
        result = _retry(dbx.files_upload_session_start, chunk)
        session_id = result.session_id
        uploaded += len(chunk)
        _print_progress(uploaded, file_size)

        # Append chunks
        while uploaded < file_size:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            cursor = UploadSessionCursor(session_id, uploaded)

            if uploaded + len(chunk) >= file_size:
                # Last chunk → finish
                commit = CommitInfo(remote_path, mode=write_mode)
                _retry(
                    dbx.files_upload_session_finish,
                    chunk, cursor, commit,
                )
            else:
                _retry(dbx.files_upload_session_append_v2, chunk, cursor)

            uploaded += len(chunk)
            _print_progress(uploaded, file_size)

    print()  # newline after progress


def _print_progress(current, total):
    pct = current / total * 100 if total > 0 else 100
    mb = current / 1e6
    total_mb = total / 1e6
    print(f"\r    {mb:,.0f} / {total_mb:,.0f} MB ({pct:.1f}%)", end="", flush=True)


def file_exists_on_dropbox(dbx, remote_path, local_size=None):
    """Check if a file already exists on Dropbox (optionally matching size)."""
    try:
        meta = dbx.files_get_metadata(remote_path)
        if local_size is not None and hasattr(meta, 'size'):
            return meta.size == local_size
        return True
    except ApiError:
        return False


# ── Main upload logic ─────────────────────────────────────────────────────────

def collect_files(local_root):
    """Collect all files under local_root with relative paths."""
    local_root = Path(local_root).resolve()
    files = []
    for dirpath, _dirnames, filenames in os.walk(local_root):
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(local_root)
            files.append((str(full), str(rel)))
    return files


def upload_folder(dbx, local_root, remote_root, skip_existing=False,
                  overwrite=False):
    """Upload an entire folder tree to Dropbox."""
    files = collect_files(local_root)
    if not files:
        print("No files found to upload.")
        return

    total_size = sum(os.path.getsize(f) for f, _ in files)
    print(f"\nFiles to upload: {len(files)}  ({total_size / 1e9:.2f} GB)")
    print(f"Local root:  {local_root}")
    print(f"Remote root: {remote_root}\n")

    write_mode = WriteMode.overwrite if overwrite else WriteMode.add

    uploaded_count = 0
    skipped_count = 0
    failed = []
    cumulative_bytes = 0

    for i, (local_path, rel_path) in enumerate(files):
        # Dropbox paths use forward slashes and must start with /
        remote_path = f"{remote_root}/{rel_path}".replace("\\", "/")
        # Ensure no double slashes
        while "//" in remote_path:
            remote_path = remote_path.replace("//", "/")

        file_size = os.path.getsize(local_path)
        size_mb = file_size / 1e6

        # Skip check
        if skip_existing and file_exists_on_dropbox(dbx, remote_path, file_size):
            print(f"  [{i+1}/{len(files)}] SKIP (exists): {rel_path}")
            skipped_count += 1
            cumulative_bytes += file_size
            continue

        print(f"  [{i+1}/{len(files)}] {rel_path}  ({size_mb:,.1f} MB)")

        try:
            if file_size <= SMALL_FILE_LIMIT:
                upload_small_file(dbx, local_path, remote_path, write_mode)
            else:
                upload_large_file(dbx, local_path, remote_path, write_mode)
            uploaded_count += 1
            cumulative_bytes += file_size
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append((rel_path, str(e)))

        # Overall progress
        overall_pct = cumulative_bytes / total_size * 100 if total_size > 0 else 100
        print(f"    Overall: {overall_pct:.1f}% ({cumulative_bytes / 1e9:.2f} / "
              f"{total_size / 1e9:.2f} GB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"  Uploaded: {uploaded_count}")
    print(f"  Skipped:  {skipped_count}")
    print(f"  Failed:   {len(failed)}")
    if failed:
        for f, e in failed:
            print(f"    ✗ {f}: {e}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload a local folder/file to Dropbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --local ./data --remote /datasets/goal1
  %(prog)s --local ./data --shared-link "https://www.dropbox.com/scl/fo/..."
  %(prog)s --local ./data --remote /data --skip-existing --token "sl.xxx"
        """,
    )
    parser.add_argument("--local", required=True,
                        help="Local folder or file to upload")
    parser.add_argument("--remote", default=None,
                        help="Dropbox destination path (e.g. /datasets/goal1)")
    parser.add_argument("--shared-link", default=None,
                        help="Dropbox shared folder link (auto-resolves to path)")
    parser.add_argument("--token", default=None,
                        help="Dropbox access token")
    parser.add_argument("--refresh-token", default=None,
                        help="Dropbox refresh token (long-lived)")
    parser.add_argument("--app-key", default=None,
                        help="Dropbox app key (used with refresh token)")
    parser.add_argument("--app-secret", default=None,
                        help="Dropbox app secret (used with refresh token)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already exist on Dropbox")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    if not os.path.exists(args.local):
        sys.exit(f"ERROR: Local path not found: {args.local}")

    dbx = get_dbx(args)

    # Resolve destination path
    if args.shared_link:
        remote_base = resolve_shared_link_path(dbx, args.shared_link)
        if remote_base is None:
            sys.exit("ERROR: Could not resolve shared link to a Dropbox path.")
        # If uploading a folder, append folder name
        if os.path.isdir(args.local):
            folder_name = os.path.basename(os.path.normpath(args.local))
            remote_root = f"{remote_base}/{folder_name}"
        else:
            remote_root = f"{remote_base}/{os.path.basename(args.local)}"
    elif args.remote:
        remote_root = args.remote
        if not remote_root.startswith("/"):
            remote_root = "/" + remote_root
    else:
        sys.exit("ERROR: Provide either --remote or --shared-link")

    print(f"Destination: {remote_root}")

    if os.path.isdir(args.local):
        upload_folder(dbx, args.local, remote_root,
                      skip_existing=args.skip_existing,
                      overwrite=args.overwrite)
    else:
        # Single file
        file_size = os.path.getsize(args.local)
        remote_path = remote_root
        write_mode = WriteMode.overwrite if args.overwrite else WriteMode.add
        print(f"Uploading: {args.local} ({file_size / 1e6:,.1f} MB)")
        if file_size <= SMALL_FILE_LIMIT:
            upload_small_file(dbx, args.local, remote_path, write_mode)
        else:
            upload_large_file(dbx, args.local, remote_path, write_mode)
        print("Done!")


if __name__ == "__main__":
    main()
