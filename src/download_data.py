"""Download FHWA NBI yearly delimited files for the configured state/years.

If download fails, prints clear manual placement instructions. Never invents data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from src.config import DATA_RAW, FHWA_DELIMITED_URL, STATE_ABBR, YEARS


def expected_filename(year: int, state: str = STATE_ABBR) -> str:
    """Return the conventional FHWA delimited filename for a state/year."""
    return f"{state.upper()}{year % 100:02d}.txt"


def expected_url(year: int, state: str = STATE_ABBR) -> str:
    """Return the FHWA delimited download URL for a state/year."""
    return FHWA_DELIMITED_URL.format(
        year=year, state=state.upper(), yy=f"{year % 100:02d}"
    )


def manual_instructions(failed: list[tuple[int, str]], raw_dir: Path) -> str:
    """Build human-readable instructions for placing files by hand."""
    lines = [
        "",
        "=" * 72,
        "DOWNLOAD FAILED — place files manually",
        "=" * 72,
        f"Put the following files into: {raw_dir.resolve()}",
        "",
        "Source pages (pick the year, then Delimited / Comma Delimited):",
        "  https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm",
        "  https://www.fhwa.dot.gov/bridge/nbi/nbi_data.cfm",
        "",
        "Direct URL pattern:",
        "  https://www.fhwa.dot.gov/bridge/nbi/{YEAR}/delimited/{STATE}{YY}.txt",
        "  Example: https://www.fhwa.dot.gov/bridge/nbi/2024/delimited/VA24.txt",
        "",
        "Required files:",
    ]
    for year, url in failed:
        lines.append(f"  - {expected_filename(year)}  <-  {url}")
    lines.extend(
        [
            "",
            "Files are comma-separated with a header row; text qualifier is a single quote.",
            "Do NOT create synthetic data. Use only official FHWA NBI files.",
            "=" * 72,
            "",
        ]
    )
    return "\n".join(lines)


def download_year(
    year: int,
    raw_dir: Path,
    state: str = STATE_ABBR,
    timeout: int = 120,
    force: bool = False,
) -> Path:
    """Download one yearly delimited file. Raises on HTTP/IO errors."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / expected_filename(year, state)
    url = expected_url(year, state)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return dest

    print(f"[get ] {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    # Basic sanity: must look like a delimited NBI file with a header.
    text = resp.content
    if len(text) < 200 or b"STRUCTURE_NUMBER" not in text[:5000]:
        raise ValueError(
            f"Downloaded content for {year} does not look like an NBI delimited file."
        )
    dest.write_bytes(text)
    print(f"[ok  ] wrote {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def download_all(
    years: tuple[int, ...] = YEARS,
    state: str = STATE_ABBR,
    raw_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Download all configured years. Returns paths; exits non-zero on failure."""
    raw_dir = raw_dir or DATA_RAW
    paths: list[Path] = []
    failed: list[tuple[int, str]] = []

    for year in years:
        try:
            paths.append(download_year(year, raw_dir, state=state, force=force))
        except Exception as exc:  # noqa: BLE001 — surface any download failure clearly
            print(f"[fail] year={year}: {exc}", file=sys.stderr)
            failed.append((year, expected_url(year, state)))

    if failed:
        print(manual_instructions(failed, raw_dir), file=sys.stderr)
        missing_on_disk = [
            y
            for y in years
            if not (raw_dir / expected_filename(y, state)).exists()
            or (raw_dir / expected_filename(y, state)).stat().st_size == 0
        ]
        if missing_on_disk:
            raise SystemExit(1)

    print(f"Download complete: {len(paths)} file(s) in {raw_dir}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download FHWA NBI delimited yearly files."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist."
    )
    parser.add_argument(
        "--state", default=STATE_ABBR, help="Two-letter state abbreviation (default VA)."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(YEARS),
        help="Inventory years to download.",
    )
    args = parser.parse_args(argv)
    download_all(years=tuple(args.years), state=args.state, force=args.force)


if __name__ == "__main__":
    main()
