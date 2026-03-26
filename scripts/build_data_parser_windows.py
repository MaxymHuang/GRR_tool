"""Build the standalone Data Parser Windows executable with PyInstaller."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = REPO_ROOT / "DataParser.spec"
DIST_DIR = REPO_ROOT / "dist" / "windows-data-parser"
BUILD_DIR = REPO_ROOT / "build" / "windows-data-parser"


def _ensure_environment() -> None:
    if sys.platform != "win32":
        print(
            "[warn] This script targets Windows. You are on %s; paths in the spec "
            "still use the repo layout — adjust or run on Windows if the build fails."
            % sys.platform,
            file=sys.stderr,
        )

    if not SPEC_FILE.exists():
        raise SystemExit(f"Missing PyInstaller spec: {SPEC_FILE}")

    entry = REPO_ROOT / "data_parser_app" / "data_parser_app" / "cli.py"
    if not entry.exists():
        raise SystemExit(f"Missing entry script: {entry}")

    if shutil.which("pyinstaller") is None:
        raise SystemExit(
            "PyInstaller not on PATH. From repo root: uv pip install pyinstaller"
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
    print(f"\nOutput: {DIST_DIR / 'DataParser.exe'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build DataParser.exe (PyInstaller)")
    parser.add_argument("--debug", action="store_true", help="Verbose PyInstaller log")
    args = parser.parse_args(argv)

    _ensure_environment()
    _build(debug=args.debug)


if __name__ == "__main__":
    main()
