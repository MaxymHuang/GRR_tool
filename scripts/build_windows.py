"""Helper script to build the Windows executable with PyInstaller.

This script expects to run on Windows with PyInstaller available in the
environment (e.g. installed via ``uv pip install pyinstaller``). It cleans any
previous build artifacts and produces the packaged application using the
existing ``GRRTool.spec`` configuration.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = REPO_ROOT / "GRRTool.spec"
DIST_DIR = REPO_ROOT / "dist" / "windows"
BUILD_DIR = REPO_ROOT / "build" / "windows"


def _ensure_environment() -> None:
    if sys.platform != "win32":
        print(
            "[warn] This build script is configured for Windows. "
            "You are running on %s." % sys.platform,
            file=sys.stderr,
        )

    if not SPEC_FILE.exists():
        raise SystemExit(f"Missing PyInstaller spec file: {SPEC_FILE}")

    if shutil.which("pyinstaller") is None:
        raise SystemExit(
            "PyInstaller is not installed. Install it with "
            "`uv pip install pyinstaller`."
        )


def _build(debug: bool) -> None:
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
    ]

    if debug:
        cmd.append("--log-level=DEBUG")

    cmd.append(str(SPEC_FILE))

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Windows executable")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable PyInstaller debug logging",
    )
    args = parser.parse_args(argv)

    _ensure_environment()
    _build(debug=args.debug)


if __name__ == "__main__":
    main()

