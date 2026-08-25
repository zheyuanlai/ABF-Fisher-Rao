#!/usr/bin/env python3
"""Audit a publication figure's required PNG/PDF output pair."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _message(report: dict[str, Any], level: str, text: str) -> None:
    report[level].append(text)


def audit_png(path: Path, report: dict[str, Any], min_dpi: int, expected_width_in: float | None) -> None:
    try:
        from PIL import Image
    except ImportError:
        _message(report, "warnings", "Pillow is unavailable; PNG metadata was not inspected.")
        return

    with Image.open(path) as image:
        width_px, height_px = image.size
        report["files"]["png"] = {
            "path": str(path),
            "width_px": width_px,
            "height_px": height_px,
            "mode": image.mode,
            "metadata_dpi": image.info.get("dpi"),
        }
        if width_px <= 0 or height_px <= 0:
            _message(report, "errors", "PNG has invalid pixel dimensions.")
        if expected_width_in:
            effective_dpi = width_px / expected_width_in
            report["files"]["png"]["effective_dpi_at_expected_width"] = round(effective_dpi, 1)
            if effective_dpi < min_dpi:
                _message(
                    report,
                    "errors",
                    f"PNG effective resolution is {effective_dpi:.1f} DPI, below {min_dpi} DPI.",
                )
        elif width_px < 1000:
            _message(
                report,
                "warnings",
                "PNG is narrower than 1000 px; provide --expected-width-in to assess effective DPI.",
            )


def _pdf_page_size(path: Path) -> tuple[int, float, float] | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(str(path))
    first = reader.pages[0]
    width_in = float(first.mediabox.width) / 72.0
    height_in = float(first.mediabox.height) / 72.0
    return len(reader.pages), width_in, height_in


def _audit_pdf_fonts(path: Path, report: dict[str, Any]) -> None:
    executable = shutil.which("pdffonts")
    if not executable:
        _message(report, "warnings", "pdffonts is unavailable; font embedding was not checked.")
        return
    completed = subprocess.run(
        [executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _message(report, "warnings", "pdffonts could not inspect the PDF.")
        return
    lines = [line.split() for line in completed.stdout.splitlines()[2:] if line.strip()]
    unembedded = [parts[0] for parts in lines if len(parts) >= 6 and parts[5].lower() != "yes"]
    if unembedded:
        _message(report, "warnings", "Unembedded PDF fonts: " + ", ".join(sorted(set(unembedded))))
    report["files"]["pdf"]["font_records"] = len(lines)


def audit_pdf(path: Path, report: dict[str, Any], expected_width_in: float | None) -> None:
    report["files"]["pdf"] = {"path": str(path), "size_bytes": path.stat().st_size}
    details = _pdf_page_size(path)
    if details is None:
        _message(report, "warnings", "pypdf is unavailable; PDF page dimensions were not inspected.")
    else:
        pages, width_in, height_in = details
        report["files"]["pdf"].update(
            {
                "pages": pages,
                "width_in": round(width_in, 3),
                "height_in": round(height_in, 3),
            }
        )
        if pages != 1:
            _message(report, "warnings", f"Expected one PDF page, found {pages}.")
        if width_in <= 0 or height_in <= 0:
            _message(report, "errors", "PDF has invalid page dimensions.")
        if expected_width_in:
            relative_error = abs(width_in - expected_width_in) / expected_width_in
            report["files"]["pdf"]["expected_width_in"] = expected_width_in
            report["files"]["pdf"]["width_relative_error"] = round(relative_error, 4)
            if relative_error > 0.02:
                _message(
                    report,
                    "warnings",
                    f"PDF width is {width_in:.3f} in, differing from the expected "
                    f"{expected_width_in:.3f} in by {100 * relative_error:.1f}%.",
                )
    _audit_pdf_fonts(path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basename", type=Path, help="Shared basename, or either the .png or .pdf path")
    parser.add_argument("--expected-width-in", type=float, default=None)
    parser.add_argument("--min-dpi", type=int, default=300)
    parser.add_argument("--json", type=Path, default=None, dest="json_path")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as a failing audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.basename.expanduser()
    if base.suffix.lower() in {".png", ".pdf"}:
        base = base.with_suffix("")
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    report: dict[str, Any] = {"basename": str(base), "files": {}, "errors": [], "warnings": []}

    for path in (png_path, pdf_path):
        if not path.is_file():
            _message(report, "errors", f"Missing required output: {path}")
    if png_path.is_file():
        audit_png(png_path, report, args.min_dpi, args.expected_width_in)
    if pdf_path.is_file():
        audit_pdf(pdf_path, report, args.expected_width_in)

    png_info = report["files"].get("png")
    pdf_info = report["files"].get("pdf")
    if png_info and pdf_info and pdf_info.get("width_in") and pdf_info.get("height_in"):
        png_ratio = png_info["width_px"] / png_info["height_px"]
        pdf_ratio = pdf_info["width_in"] / pdf_info["height_in"]
        ratio_error = abs(png_ratio - pdf_ratio) / pdf_ratio
        report["pair_aspect_ratio_relative_error"] = round(ratio_error, 5)
        if ratio_error > 0.01:
            _message(report, "warnings", "PNG and PDF aspect ratios differ by more than 1%.")

    report["status"] = "fail" if report["errors"] or (args.strict and report["warnings"]) else "pass"
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
