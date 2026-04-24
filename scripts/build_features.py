from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.engineer import build_features


RAW_DIR = ROOT / "data" / "raw"
FEATURE_DIR = ROOT / "data" / "features"


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        raise SystemExit(f"No parquet files found in {RAW_DIR}")

    for raw_file in raw_files:
        frame = pd.read_parquet(raw_file)
        features = build_features(frame)
        output = FEATURE_DIR / f"{raw_file.stem}_features.parquet"
        features.to_parquet(output)
        print(f"Wrote {output.relative_to(ROOT)} rows={len(features)} columns={len(features.columns)}")


if __name__ == "__main__":
    main()
