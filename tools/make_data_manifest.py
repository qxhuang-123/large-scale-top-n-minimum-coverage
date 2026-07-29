"""Create a reproducibility manifest without copying large research data."""

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in sorted(path for path in root.rglob("*") if path.is_file()):
            writer.writerow({"path": path.relative_to(root).as_posix(),
                             "bytes": path.stat().st_size, "sha256": sha256(path)})


if __name__ == "__main__":
    main()
