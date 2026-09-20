from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.cleaning import clean_all
from src.config import RAW_DIR
from src.modeling import train_all

if __name__=="__main__":
    print(train_all(clean_all(RAW_DIR)))

