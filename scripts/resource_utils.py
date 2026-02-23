#!/usr/bin/env python3
"""
resource_utils.py — Detect CPU / RAM / GPU resources for safe parallelism.

Handles SLURM cgroup memory limits (common on HPC clusters).
"""

import os


def get_rss_mb():
    """Return current process RSS in MB, or -1 if unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return -1


def get_cgroup_mem_limit_gb():
    """Return the cgroup memory limit in GB, or None if not in a cgroup.

    Checks both cgroup v1 and v2 paths.  Falls back to system RAM.
    """
    # cgroup v1: read cgroup path from /proc/self/cgroup
    try:
        cgroup_path = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3 and parts[1] == "memory":
                    cgroup_path = parts[2]
                    break

        if cgroup_path:
            # Try job-level limit first (SLURM sets this)
            # Walk up from task_0 → step_batch → job_XXXXX
            parts = cgroup_path.rstrip("/").split("/")
            for depth in range(len(parts), 1, -1):
                candidate = "/".join(parts[:depth])
                limit_file = f"/sys/fs/cgroup/memory{candidate}/memory.limit_in_bytes"
                if os.path.exists(limit_file):
                    with open(limit_file) as f:
                        val = int(f.read().strip())
                    # Ignore very large values (effectively unlimited)
                    if val < 2**62:
                        return round(val / (1024**3), 1)
    except Exception:
        pass

    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
        if val != "max":
            return round(int(val) / (1024**3), 1)
    except Exception:
        pass

    return None


def get_cgroup_mem_usage_gb():
    """Return current cgroup memory usage in GB, or None."""
    try:
        cgroup_path = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3 and parts[1] == "memory":
                    cgroup_path = parts[2]
                    break
        if cgroup_path:
            parts = cgroup_path.rstrip("/").split("/")
            for depth in range(len(parts), 1, -1):
                candidate = "/".join(parts[:depth])
                usage_file = f"/sys/fs/cgroup/memory{candidate}/memory.usage_in_bytes"
                if os.path.exists(usage_file):
                    with open(usage_file) as f:
                        val = int(f.read().strip())
                    return round(val / (1024**3), 1)
    except Exception:
        pass
    return None


def detect_resources():
    """Detect CPU cores, RAM, GPU availability.

    Uses cgroup memory limit when available (SLURM jobs).
    """
    info = {"cpu_count": os.cpu_count() or 1}

    # RAM from /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_total_gb"] = round(int(line.split()[1]) / (1024 ** 2), 1)
                elif line.startswith("MemAvailable:"):
                    info["ram_available_gb"] = round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        info["ram_total_gb"] = 0
        info["ram_available_gb"] = 0

    # Override with cgroup limit if present (SLURM)
    cgroup_limit = get_cgroup_mem_limit_gb()
    cgroup_usage = get_cgroup_mem_usage_gb()
    if cgroup_limit is not None:
        info["cgroup_limit_gb"] = cgroup_limit
        info["cgroup_usage_gb"] = cgroup_usage
        # Use cgroup limit as effective available RAM
        if cgroup_usage is not None:
            info["ram_available_gb"] = round(cgroup_limit - cgroup_usage, 1)
        else:
            info["ram_available_gb"] = round(cgroup_limit * 0.8, 1)
        info["ram_total_gb"] = cgroup_limit

    # GPU via torch
    info["gpu_available"] = False
    info["torch_cuda"] = False
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_cuda"] = True
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        pass

    return info


def compute_safe_jobs(ram_available_gb, cpu_count, mem_per_job_gb=3.0):
    """Return a safe number of parallel workers given available resources."""
    ram_jobs = max(1, int(ram_available_gb // mem_per_job_gb))
    cpu_jobs = max(1, cpu_count - 1)
    return min(ram_jobs, cpu_jobs)
