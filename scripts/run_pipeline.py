from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.cleaning import clean_all
from src.database import persist_tables
from src.modeling import train_all


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--raw-dir",default=ROOT/"data"/"raw"); parser.add_argument("--skip-training",action="store_true"); args=parser.parse_args()
    print("[1/3] Cleaning all raw datasets..."); tables=clean_all(args.raw_dir)
    print("[2/3] Writing processed CSV files and development database..."); persist_tables(tables)
    print("[3/3] Training models..." if not args.skip_training else "[3/3] Model training skipped")
    if not args.skip_training:
        metrics=train_all(tables); print(json.dumps(metrics,indent=2,default=float))
    print("Pipeline completed successfully.")

if __name__=="__main__": main()

