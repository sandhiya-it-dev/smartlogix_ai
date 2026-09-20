from __future__ import annotations

import argparse, random
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parents[1]
CLASSES=["healthy","broken_propeller","frame_damage","motor_damage","landing_gear_damage","battery_damage"]


def draw_drone(seed:int,kind:str,size:int=416):
    rng=random.Random(seed); img=np.full((size,size,3),rng.randint(205,240),np.uint8); c=size//2
    cv2.rectangle(img,(c-55,c-38),(c+55,c+38),(70,75,80),-1)
    motors=[]
    for dx,dy in [(-115,-105),(115,-105),(-115,105),(115,105)]:
        cv2.line(img,(c+(-45 if dx<0 else 45),c+(-28 if dy<0 else 28)),(c+dx,c+dy),(45,45,45),10)
        cv2.circle(img,(c+dx,c+dy),24,(40,40,45),-1); cv2.ellipse(img,(c+dx,c+dy),(50,10),rng.randint(0,170),0,360,(100,105,110),5); motors.append((c+dx,c+dy))
    box=None
    if kind=="broken_propeller":
        x,y=motors[0]; cv2.line(img,(x-42,y),(x-8,y),(0,0,230),9); box=(x-60,y-35,x+60,y+35)
    elif kind=="frame_damage":
        cv2.line(img,(c-20,c-38),(c+12,c+38),(0,0,230),7); box=(c-58,c-45,c+58,c+45)
    elif kind=="motor_damage":
        x,y=motors[1]; cv2.circle(img,(x,y),30,(0,0,230),6); box=(x-38,y-38,x+38,y+38)
    elif kind=="landing_gear_damage":
        cv2.line(img,(c-50,c+38),(c-75,c+78),(0,0,230),8); box=(c-90,c+28,c-35,c+90)
    elif kind=="battery_damage":
        cv2.rectangle(img,(c-34,c-25),(c+34,c+25),(0,0,230),6); box=(c-42,c-33,c+42,c+33)
    return img,box


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--samples-per-class",type=int,default=80); ap.add_argument("--output",type=Path,default=ROOT/"data"/"cv"); args=ap.parse_args()
    random.seed(42)
    for split in ["train","val"]:
        (args.output/"images"/split).mkdir(parents=True,exist_ok=True); (args.output/"labels"/split).mkdir(parents=True,exist_ok=True)
    for class_id,kind in enumerate(CLASSES):
        for i in range(args.samples_per_class):
            split="val" if i%5==0 else "train"; img,box=draw_drone(class_id*10000+i,kind); name=f"{kind}_{i:04d}"
            cv2.imwrite(str(args.output/"images"/split/f"{name}.jpg"),img)
            label=""
            if box:
                x1,y1,x2,y2=box; label=f"{class_id-1} {(x1+x2)/2/416:.6f} {(y1+y2)/2/416:.6f} {(x2-x1)/416:.6f} {(y2-y1)/416:.6f}\n"
            (args.output/"labels"/split/f"{name}.txt").write_text(label)
    print(f"Generated synthetic CV data at {args.output.resolve()}")

if __name__=="__main__":main()

