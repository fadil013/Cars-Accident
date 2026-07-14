"""
Single-image accident check (appearance-based).

A still image has NO motion, so the video detector's speed/deceleration cues do not
apply. On one frame the honest signal is geometric: two vehicles whose boxes are in
contact (overlapping) plus, often, a person standing in the roadway beside them.
This flags a POSSIBLE collision for human review; it cannot measure speed or confirm
impact the way the video pipeline does.

Usage:
    python detect_image.py path/to/image.jpg
"""
import os, sys
import numpy as np, cv2
from ultralytics import YOLO

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT     = os.path.join(ROOT, "out")
WEIGHTS = os.path.join(os.path.dirname(ROOT), "yolo11x.pt")
VEHICLE = {1, 2, 3, 5, 7}
PERSON  = 0
NAME    = {0: "person", 1: "bike", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(path):
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(WEIGHTS)
    img = cv2.imread(path)
    if img is None:
        print(f"!! cannot read {path}"); return
    H, W = img.shape[:2]
    r = model.predict(img, imgsz=1280, conf=0.25, classes=[0, 1, 2, 3, 5, 7], verbose=False)[0]

    veh, ppl = [], []
    for b in r.boxes:
        c = int(b.cls); cf = float(b.conf); xy = b.xyxy.cpu().numpy()[0].astype(float)
        (veh if c in VEHICLE else ppl).append((c, cf, xy))

    # collision = a pair of vehicles in contact (boxes overlap)
    pairs = [(i, j, iou(veh[i][2], veh[j][2]))
             for i in range(len(veh)) for j in range(i+1, len(veh))
             if iou(veh[i][2], veh[j][2]) > 0.03]

    vis = img.copy()
    for c, cf, xy in veh:
        x1, y1, x2, y2 = map(int, xy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, f"{NAME[c]} {cf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)
    for c, cf, xy in ppl:
        x1, y1, x2, y2 = map(int, xy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(vis, "person", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2, cv2.LINE_AA)

    person_near = False
    for i, j, ov in pairs:
        a, b = veh[i][2], veh[j][2]
        ux1, uy1 = int(min(a[0], b[0])), int(min(a[1], b[1]))
        ux2, uy2 = int(max(a[2], b[2])), int(max(a[3], b[3]))
        cv2.rectangle(vis, (ux1-16, uy1-16), (ux2+16, uy2+16), (0, 0, 255), 4)
        cv2.putText(vis, "COLLISION", (ux1-16, uy1-24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cx, cy = (ux1+ux2)/2, (uy1+uy2)/2
        for _, _, pxy in ppl:
            px, py = (pxy[0]+pxy[2])/2, (pxy[1]+pxy[3])/2
            if ux1-60 < px < ux2+60 and uy1-60 < py < uy2+60:
                person_near = True

    if pairs:
        note = " + person in road" if (person_near or ppl) else ""
        cv2.rectangle(vis, (0, 0), (W, 50), (0, 0, 200), -1)
        cv2.putText(vis, f"ACCIDENT DETECTED   Collision: {len(pairs)} vehicle(s) in contact{note}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(vis, (0, 0), (430, 34), (35, 35, 35), -1)
        cv2.putText(vis, "NO COLLISION DETECTED", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

    cv2.rectangle(vis, (0, H-28), (W, H), (0, 0, 0), -1)
    cv2.putText(vis, "Single-image appearance check (no motion cues)  |  flag for human review",
                (14, H-9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    out = os.path.join(OUT, os.path.splitext(os.path.basename(path))[0] + "_detected.jpg")
    cv2.imwrite(out, vis)
    print(f"vehicles={len(veh)}  persons={len(ppl)}  collisions={len(pairs)}")
    print(f"  -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python detect_image.py <image>"); sys.exit(1)
    main(sys.argv[1])
