"""Download and verify the Amazon ESCI (Shopping Queries) dataset files.

Only the two parquet files are needed: the examples file (queries, labels, split flags) and the
products file (titles, descriptions, brands). The sources file is not used by this project.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ESCI_BASE_URL = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
CHECKSUMS_NAME = "CHECKSUMS.json"

# A Git LFS pointer is a tiny text file that starts with this marker. Getting one instead of the
# real parquet file means the download did not resolve to the actual data.
_LFS_MARKER = b"version https://git-lfs"


@dataclass(frozen=True)
class EsciFile:
    name: str
    required_columns: tuple[str, ...]


ESCI_FILES: tuple[EsciFile, ...] = (
    EsciFile(
        "shopping_queries_dataset_examples.parquet",
        (
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
            "split",
        ),
    ),
    EsciFile(
        "shopping_queries_dataset_products.parquet",
        (
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        ),
    ),
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _show_progress(name: str, done: int, total: int) -> None:
    if total:
        message = f"\r{name}: {done / 1e6:.1f} / {total / 1e6:.1f} MB"
    else:
        message = f"\r{name}: {done / 1e6:.1f} MB"
    sys.stderr.write(message)
    sys.stderr.flush()


def download_file(
    url: str, dest: Path, *, chunk_size: int = 1 << 20, progress: bool = True
) -> None:
    """Stream url to dest, via a .part file so a failed download leaves no partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "retailmind-downloader"})
    # A fresh opener per call, so proxy settings from the environment are read at call time.
    opener = urllib.request.build_opener()
    try:
        with opener.open(request, timeout=60) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(chunk_size):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    _show_progress(dest.name, done, total)
        part.replace(dest)
        if progress:
            sys.stderr.write("\n")
    finally:
        part.unlink(missing_ok=True)


def verify_parquet(path: Path, required_columns: tuple[str, ...]) -> int:
    """Check that path is a real parquet file with the expected columns. Returns its row count."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "rb") as handle:
        head = handle.read(64)
    if head.startswith(_LFS_MARKER):
        raise ValueError(
            f"{path.name} is a Git LFS pointer, not the real data. "
            "Download the file from the repository page in a browser instead."
        )
    # Open the file ourselves so it is closed before we return. On Windows an open file cannot
    # be deleted, and pq.ParquetFile(path) would keep it open until garbage collection.
    try:
        with open(path, "rb") as source:
            parquet = pq.ParquetFile(source)
            names = parquet.schema_arrow.names
            rows = parquet.metadata.num_rows
    except Exception as error:
        raise ValueError(f"{path.name} is not a readable parquet file: {error}") from error
    missing = [c for c in required_columns if c not in names]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    return rows


def _describe(path: Path, spec: EsciFile, downloaded: bool) -> dict:
    rows = verify_parquet(path, spec.required_columns)
    return {
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": sha256_file(path),
        "downloaded": downloaded,
    }


def _write_checksums(raw_dir: Path, summary: dict[str, dict]) -> None:
    (raw_dir / CHECKSUMS_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def download_all(
    raw_dir: str | Path,
    *,
    base_url: str = ESCI_BASE_URL,
    files: tuple[EsciFile, ...] = ESCI_FILES,
    force: bool = False,
    progress: bool = True,
) -> dict[str, dict]:
    """Download any missing ESCI files into raw_dir, verify them, and record checksums."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for spec in files:
        dest = raw_dir / spec.name
        downloaded = force or not dest.exists()
        if downloaded:
            download_file(f"{base_url.rstrip('/')}/{spec.name}", dest, progress=progress)
        try:
            summary[spec.name] = _describe(dest, spec, downloaded)
        except ValueError:
            if downloaded:
                dest.unlink(missing_ok=True)  # do not keep a bad download around
            raise
    _write_checksums(raw_dir, summary)
    return summary


def verify_all(
    raw_dir: str | Path, *, files: tuple[EsciFile, ...] = ESCI_FILES
) -> dict[str, dict]:
    """Verify files already in raw_dir without touching the network, and record checksums."""
    raw_dir = Path(raw_dir)
    summary = {spec.name: _describe(raw_dir / spec.name, spec, False) for spec in files}
    _write_checksums(raw_dir, summary)
    return summary
