from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report import build_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown table from scaffold result JSON files.")
    parser.add_argument("kind", choices=["eval", "speed"])
    parser.add_argument("--filename", default=None, help="Optional output filename")
    parser.add_argument("--result-dir", default=None, help="Override result directory")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--columns", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_table(
        kind=args.kind,
        columns=args.columns,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        filename=args.filename,
    )
    print(output_path)


if __name__ == "__main__":
    main()
