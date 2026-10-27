"""
Package the complete, reproducible research project into a clean ZIP archive.
Ensures secrets (.env), caches (__pycache__, .pytest_cache), and temporary files are excluded.
Usage:
    python scripts/create_zip.py
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

ZIP_NAME = "Joint_Input_Robustness_3_Condition_Experiment.zip"
TARGET_ZIP_PROJECT = PROJECT_ROOT / ZIP_NAME
TARGET_ZIP_WORKSPACE = WORKSPACE_ROOT / ZIP_NAME

EXCLUDE_PATTERNS = [
    ".env",
    "*.env",
    ".env.local",
    "*.zip",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".git",
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp"
]


def should_exclude(rel_path: Path) -> bool:
    """Check if relative path matches any exclusion pattern."""
    path_str = str(rel_path).replace("\\", "/")
    parts = path_str.split("/")

    for part in parts:
        if part in [".env", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"]:
            return True
        if part.endswith(".env"):
            return True

    for p in EXCLUDE_PATTERNS:
        if p.startswith("*") and path_str.endswith(p[1:]):
            return True

    return False


def create_archive():
    print(f"[*] Packaging project: {PROJECT_ROOT.name} -> {ZIP_NAME}")

    files_added = 0
    total_bytes = 0

    with zipfile.ZipFile(TARGET_ZIP_PROJECT, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if not should_exclude(Path(d))]

            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(PROJECT_ROOT)

                if should_exclude(rel_path):
                    continue

                arcname = f"Joint_Input_Robustness/{rel_path}"
                zipf.write(full_path, arcname)
                files_added += 1
                total_bytes += full_path.stat().st_size

    # Also copy or write to workspace root
    import shutil
    shutil.copy2(TARGET_ZIP_PROJECT, TARGET_ZIP_WORKSPACE)

    print(f"[+] Archive created successfully!")
    print(f"    Total files archived:  {files_added}")
    print(f"    Uncompressed size:     {total_bytes / 1024:.1f} KB")
    print(f"    Project ZIP location:  {TARGET_ZIP_PROJECT}")
    print(f"    Workspace ZIP copy:    {TARGET_ZIP_WORKSPACE}")


if __name__ == "__main__":
    create_archive()
