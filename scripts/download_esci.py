"""Download the ESCI dataset into data/raw/ and verify it.

Usage:
    python scripts/download_esci.py                 download missing files, verify, record checksums
    python scripts/download_esci.py --force         download again even if the files exist
    python scripts/download_esci.py --verify-only   no network; check files already in data/raw/
"""

import argparse
import sys
from pathlib import Path

from retailmind.config import load_config
from retailmind.data.esci import ESCI_BASE_URL, download_all, verify_all

MANUAL_HELP = """
If the download keeps failing, get the files by hand:
  1. Open https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset
  2. Download shopping_queries_dataset_examples.parquet and
     shopping_queries_dataset_products.parquet into {raw_dir}
  3. Run: python scripts/download_esci.py --verify-only
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if files exist")
    parser.add_argument("--verify-only", action="store_true", help="skip the download")
    parser.add_argument("--raw-dir", help="override paths.raw_dir from the config")
    parser.add_argument("--base-url", default=ESCI_BASE_URL, help="override the download URL")
    args = parser.parse_args(argv)

    cfg = load_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.resolve(cfg.paths.raw_dir)

    try:
        if args.verify_only:
            summary = verify_all(raw_dir)
        else:
            summary = download_all(raw_dir, base_url=args.base_url, force=args.force)
    except (OSError, ValueError) as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        print(MANUAL_HELP.format(raw_dir=raw_dir), file=sys.stderr)
        return 1

    print(f"\nESCI files in {raw_dir}:")
    for name, info in summary.items():
        status = "downloaded" if info["downloaded"] else "already present"
        print(
            f"  {name}\n"
            f"    {info['bytes'] / 1e6:.1f} MB, {info['rows']:,} rows, "
            f"sha256 {info['sha256'][:16]}..., {status}"
        )
    print("Checksums saved to CHECKSUMS.json in that folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
