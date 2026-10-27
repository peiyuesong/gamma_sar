#!/usr/bin/env python
"""Rebuild the gamma_sar preprint PDF from ``preprint.md``.

Pipeline (fully offline, no LaTeX installation required):

1. ``pandoc``  markdown -> standalone HTML5, MathML, figures embedded as base64
2. ``msedge``  headless print-to-PDF (A4; page margins and typography from
   ``tools/preprint.css``)

The markdown is the single source of truth. Figures are referenced inline in
``preprint.md`` as ``![](figures/figN_*_v2.png)`` so that a rebuild cannot
silently drop them.

Usage (repository root, mleDSar interpreter)::

    python tools/build_preprint.py
    python tools/build_preprint.py --out preprint_v0.2_draft.pdf --keep-html
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "preprint.md"
CSS = Path(__file__).resolve().parent / "preprint.css"

PANDOC_CANDIDATES = [
    "pandoc",
    r"C:\ANACONDA\Library\bin\pandoc.exe",
]

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_pandoc() -> str:
    for candidate in PANDOC_CANDIDATES:
        found = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if found and Path(found).exists():
            return found
    sys.exit("pandoc not found (install pandoc or adjust PANDOC_CANDIDATES)")


def find_browser() -> str:
    for candidate in EDGE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    sys.exit("no Chromium-based browser found for print-to-PDF")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "preprint_v0.2_draft.pdf"),
        help="output PDF (default: preprint_v0.2_draft.pdf in the repository root)",
    )
    parser.add_argument("--keep-html", action="store_true", help="keep build/preprint.html")
    args = parser.parse_args()

    if not MD.exists():
        sys.exit(f"manuscript not found: {MD}")

    pandoc = find_pandoc()
    browser = find_browser()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    html = build / "preprint.html"
    out = Path(args.out).resolve()

    print("[1/3] pandoc -> HTML")
    subprocess.run(
        [
            pandoc,
            str(MD),
            "-f", "markdown",
            "-t", "html5",
            "-s",
            "--mathml",
            "--embed-resources",
            "--css", str(CSS),
            "-o", str(html),
        ],
        check=True,
    )

    print("[2/3] headless print -> PDF")
    profile = Path(tempfile.gettempdir()) / "gamma_sar_print_profile"
    profile.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={out}",
            html.as_uri(),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not out.exists():
        sys.exit(f"PDF was not produced at {out}")

    print(f"[3/3] done: {out.name} ({out.stat().st_size:,} bytes)")
    if not args.keep_html:
        html.unlink(missing_ok=True)
        try:
            build.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
