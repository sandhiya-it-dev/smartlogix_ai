from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=25); ap.add_argument("--model",default="yolov8n.pt"); args=ap.parse_args()
    try: from ultralytics import YOLO
    except ImportError: raise SystemExit("Install optional dependency first: pip install ultralytics")
    root=Path(__file__).resolve().parents[1]
    YOLO(args.model).train(data=str(root/"config"/"drone_damage.yaml"),epochs=args.epochs,imgsz=416,project=str(root/"models"/"yolo"),name="drone_damage",seed=42)

if __name__=="__main__":main()

