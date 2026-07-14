"""
Accident detection for fixed traffic cameras (two-pass).

Goal: track every vehicle, and mark the accident the exact second it happens, with a red
box on every affected vehicle, no false alarms, no faked numbers.

Why two passes
--------------
A live single-pass detector cannot confirm a crash until the aftermath (the cars must be
seen to stop), which makes the alert appear late. On a recorded clip we can do better:

  PASS 1  detect + track every frame, and record candidate impacts (two vehicles overlap
          while at least one violently sheds speed), plus fire and hard-stop candidates.
  CONFIRM keep only candidates that actually became a wreck (an involved vehicle comes to
          rest and stays), which throws out passing-traffic overlaps and brake taps.
  PASS 2  re-render from the impact frame, so the red alert appears the instant it occurs
          and every vehicle in the pile-up gets its own red box.

All cues are explainable motion/appearance; confidence is a heuristic cue score in [0,1],
NOT an accuracy figure. Human-in-the-loop flagger.

Usage:
    python detect_accidents.py                       # runs the three bundled clips
    python detect_accidents.py "Car Accident 1.0.mov"
"""
import os, sys, json, math, subprocess
from collections import deque, defaultdict
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------------------- config -----------------------------
ROOT      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(ROOT, "out")
WEIGHTS   = os.path.join(os.path.dirname(ROOT), "yolo11x.pt")
IMGSZ     = 1280
CONF      = 0.25
DEVICE    = 0
TRACK_CLS = [0, 1, 2, 3, 5, 7]
VEHICLE   = {1, 2, 3, 5, 7}
PERSON    = 0

VEL_WIN      = 5
V_MOVE       = 2.6
V_STOP       = 1.1
V_WASMOVING  = 4.0
DECEL_WIN    = 8
SHOCK_FRAC   = 0.35        # a crash sheds > 65% of its speed within DECEL_WIN
IOU_COLLIDE  = 0.08        # light box overlap = first contact (oblique cars touch side-by-side)
IOU_HOLD     = 2           # frames of overlap before a pair is a candidate
STOP_HOLD    = 12
CONFIRM_WIN  = 120         # look-ahead frames to confirm a candidate became a wreck (cars slide first)
CONFIRM_STOP = 12          # an involved vehicle must come to a real halt (rejects brief traffic slows)
FOLD_FRAMES  = 72          # candidates within this time+space are one pile-up
FOLD_PX      = 500         # a separate overlap far away is a different event, not this wreck
SPIN_DEG     = 70.0
NEAR_RADIUS  = 260

FIRE_MINAREA = 1400
FIRE_HOTMIN  = 25
FIRE_CELL    = 96
FIRE_SEEN    = 7
FIRE_WIN     = 11


# ----------------------------- helpers -----------------------------
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def fire_components(frame):
    b, g, r = cv2.split(frame.astype(np.int16))
    fire = ((r > 160) & (g > 55) & (g < r) & (b < g) & ((r - b) > 55)).astype(np.uint8)
    hot = ((r > 228) & (g > 185)).astype(np.uint8)
    fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fire = cv2.dilate(fire, np.ones((5, 5), np.uint8), 1)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fire, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < FIRE_MINAREA: continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if int(hot[y:y+h, x:x+w].sum()) < FIRE_HOTMIN: continue
        out.append((int(cent[i][0]), int(cent[i][1]), area, (x, y, x+w, y+h)))
    return out


def global_shift(prev_gray, gray):
    if prev_gray is None: return 0.0, 0.0
    (dx, dy), _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(gray))
    if abs(dx) > 8 or abs(dy) > 8: return 0.0, 0.0
    return dx, dy


def ang_diff(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180 else 360 - d


def reason_of(kinds):
    if "FIRE" in kinds: return "Vehicle fire"
    if "COLLISION" in kinds: return "Collision" + (" + spinout" if "SPINOUT" in kinds else "")
    if "IMPACT" in kinds: return "Crash: violent impact + rotation"
    if "SUDDEN_STOP" in kinds:
        if "PERSON_IN_ROAD" in kinds: return "Crash: sudden stop, person down"
        return "Crash: sudden stop + traffic disruption"
    return "Traffic anomaly"


def cluster(records):
    """Fold raw per-frame detections into events by time + space proximity.
    `involved[id]` records the frame each vehicle first joined the crash, so in the
    render each car turns red at ITS moment of contact, not all at once."""
    events = []
    for rec in sorted(records, key=lambda r: r["fi"]):
        cx, cy = (rec["bbox"][0]+rec["bbox"][2])/2, (rec["bbox"][1]+rec["bbox"][3])/2
        host = None
        for e in events:
            ecx, ecy = (e["bbox"][0]+e["bbox"][2])/2, (e["bbox"][1]+e["bbox"][3])/2
            if (rec["fi"] - e["last_fi"] < FOLD_FRAMES and math.hypot(cx-ecx, cy-ecy) < FOLD_PX) \
               or (rec["ids"] & e["ids"]):
                host = e; break
        if host is None:
            host = {"impact": rec["onset"], "prov": rec["fi"], "last_fi": rec["fi"],
                    "ids": set(rec["ids"]), "kinds": set(rec["kinds"]), "bbox": rec["bbox"], "involved": {}}
            events.append(host)
        else:
            host["ids"] |= rec["ids"]; host["kinds"] |= rec["kinds"]
            host["impact"] = min(host["impact"], rec["onset"])
            host["last_fi"] = rec["fi"]; host["bbox"] = rec["bbox"]
        for i in rec["ids"]:
            host["involved"][i] = min(host["involved"].get(i, rec["onset"]), rec["onset"])
    return events


# ----------------------------- pass 1: detect + collect -----------------------------
def collect(video):
    model = YOLO(WEIGHTS)
    tracks, pairs = {}, {}
    fire_hist = defaultdict(lambda: deque(maxlen=FIRE_WIN)); fire_first = {}
    prev_gray = None; cam = np.array([0.0, 0.0])
    fdata = []                       # per frame: [(id, cls, bbox, stopped, sm)]
    stop_log = defaultdict(dict)     # id -> {fi: stopped}
    raw_coll, raw_fire, raw_impact = [], [], []
    seen_ids = set()

    stream = model.track(source=video, stream=True, tracker="bytetrack.yaml",
                         classes=TRACK_CLS, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)
    for fi, r in enumerate(stream):
        frame = r.orig_img
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.25, fy=0.25)
        dx, dy = global_shift(prev_gray, gray); cam += np.array([dx*4, dy*4]); prev_gray = gray

        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy(); ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for bb, tid, c in zip(xyxy, ids, cls):
                dets.append((int(tid), int(c), bb.astype(float),
                             float((bb[0]+bb[2])/2), float((bb[1]+bb[3])/2)))

        perframe = []
        persons = []
        for tid, c, bb, cx, cy in dets:
            if c in VEHICLE: seen_ids.add(tid)
            if c == PERSON: persons.append((cx, cy))
            comp = np.array([cx, cy]) - cam
            st = tracks.get(tid) or {"pos": deque(maxlen=40), "spd": deque(maxlen=40),
                    "sm": deque(maxlen=40), "maxspd": 0.0, "stopped": 0, "head": None,
                    "decel_f": -999, "decel_start": -999}
            tracks[tid] = st; st["pos"].append(comp); st["bb"] = bb; st["cls"] = c
            if len(st["pos"]) > VEL_WIN:
                d = st["pos"][-1] - st["pos"][-1-VEL_WIN]
                spd = float(np.hypot(*d)) / VEL_WIN
                head = math.degrees(math.atan2(d[1], d[0])) if spd > 0.6 else st["head"]
            else:
                spd, head = 0.0, st["head"]
            st["spd"].append(spd); sm = float(np.mean(list(st["spd"])[-3:]))
            st["sm"].append(sm); st["maxspd"] = max(st["maxspd"], sm)
            smrec = list(st["sm"]); peak = max(smrec[-DECEL_WIN:]) if len(smrec) >= DECEL_WIN else sm
            was_moving = st["maxspd"] > V_WASMOVING
            prev_sm = smrec[-2] if len(smrec) >= 2 else sm
            if was_moving and prev_sm >= 0.8*peak and sm < 0.8*peak:
                st["decel_start"] = fi
            shock = was_moving and peak > V_WASMOVING*0.8 and sm < SHOCK_FRAC*peak
            if shock: st["decel_f"] = fi
            spin = (head is not None and st["head"] is not None and spd > 0.6
                    and ang_diff(head, st["head"]) > SPIN_DEG and (fi - st["decel_f"]) < 15)
            if head is not None: st["head"] = head
            st["stopped"] = st["stopped"] + 1 if (sm < V_STOP and was_moving) else 0
            st["spin_f"] = fi if spin else st.get("spin_f", -999)

            perframe.append((tid, c, bb, st["stopped"], sm))
            if c in VEHICLE: stop_log[tid][fi] = st["stopped"]
            # single-vehicle violent impact (shock + rotation)
            if c in VEHICLE and shock and spin:
                raw_impact.append({"fi": fi, "onset": max(0, st["decel_start"] if st["decel_start"] > 0 else fi),
                                   "ids": {tid}, "kinds": {"IMPACT", "SPINOUT"}, "bbox": list(bb)})
        fdata.append(perframe)

        # collision candidates: an overlap run that shocks; backdate to first contact of the run
        veh = [(tid, bb) for tid, c, bb, cx, cy in dets if c in VEHICLE]
        for i in range(len(veh)):
            for j in range(i+1, len(veh)):
                a, bA = veh[i]; b, bB = veh[j]; pr = (a, b) if a < b else (b, a)
                if iou(bA, bB) > IOU_COLLIDE:
                    p = pairs.setdefault(pr, {"ov": 0, "start": fi, "shock_f": None})
                    p["ov"] += 1
                    if (fi - tracks.get(a, {}).get("decel_f", -999) < 15 or
                            fi - tracks.get(b, {}).get("decel_f", -999) < 15):
                        if p["shock_f"] is None: p["shock_f"] = fi
                    if p["ov"] >= IOU_HOLD and p["shock_f"] is not None:
                        onset = max(p["start"], p["shock_f"] - 24)      # first contact, up to ~1s before the shock
                        bx = [min(bA[0], bB[0]), min(bA[1], bB[1]), max(bA[2], bB[2]), max(bA[3], bB[3])]
                        kinds = {"COLLISION"}
                        if fi - tracks.get(a, {}).get("spin_f", -999) < 15 or fi - tracks.get(b, {}).get("spin_f", -999) < 15:
                            kinds.add("SPINOUT")
                        raw_coll.append({"fi": fi, "onset": onset, "ids": {a, b}, "kinds": kinds, "bbox": bx})
                elif pr in pairs:
                    del pairs[pr]                                        # separated -> reset the overlap run

        # fire candidates
        for cx, cy, area, bb in fire_components(frame):
            cell = (cx//FIRE_CELL, cy//FIRE_CELL); fire_first.setdefault(cell, fi)
            fire_hist[cell].append((fi, area))
            rec = [(f, a) for f, a in fire_hist[cell] if fi - f < FIRE_WIN]
            areas = np.array([a for _, a in rec], dtype=float)
            flick = float(areas.std()/(areas.mean()+1e-6)) if len(areas) >= 3 else 0.0
            grow = float(areas.max()/(areas.min()+1e-6)) if len(areas) >= 3 else 1.0
            if len(rec) >= FIRE_SEEN and fire_first[cell] > 8 and (flick > 0.22 or grow > 1.8):
                raw_fire.append({"fi": fi, "onset": fire_first[cell], "ids": set(),
                                 "kinds": {"FIRE"}, "bbox": list(bb)})

    return dict(fdata=fdata, stop_log=stop_log, raw_coll=raw_coll, raw_fire=raw_fire,
                raw_impact=raw_impact, seen_ids=seen_ids)


# ----------------------------- confirm -----------------------------
def confirm_events(data, fps):
    events = []
    # collisions: keep only those where an involved vehicle comes to rest afterwards
    for e in cluster(data["raw_coll"]):
        ok = any(data["stop_log"].get(i, {}).get(f, 0) >= CONFIRM_STOP
                 for i in e["ids"] for f in range(e["prov"], e["prov"] + CONFIRM_WIN))
        if ok: events.append(e)
    # single-vehicle violent impacts: same rest test
    for e in cluster(data["raw_impact"]):
        ok = any(data["stop_log"].get(i, {}).get(f, 0) >= CONFIRM_STOP
                 for i in e["ids"] for f in range(e["prov"], e["prov"] + CONFIRM_WIN))
        if ok: events.append(e)
    # fire: temporally self-confirmed
    events += cluster(data["raw_fire"])

    # fold across types, earliest impact wins
    events.sort(key=lambda e: e["impact"])
    merged = []
    for e in events:
        cx, cy = (e["bbox"][0]+e["bbox"][2])/2, (e["bbox"][1]+e["bbox"][3])/2
        host = next((m for m in merged
                     if abs(e["impact"] - m["impact"]) < FOLD_FRAMES
                     and (math.hypot(cx-(m["bbox"][0]+m["bbox"][2])/2, cy-(m["bbox"][1]+m["bbox"][3])/2) < FOLD_PX
                          or e["ids"] & m["ids"])), None)
        if host:
            host["ids"] |= e["ids"]; host["kinds"] |= e["kinds"]; host["impact"] = min(host["impact"], e["impact"])
            for k, v in e.get("involved", {}).items():
                host["involved"][k] = min(host["involved"].get(k, v), v)
        else: merged.append(e)
    for m in merged:
        m["first_t"] = round(m["impact"] / fps, 2)
        m["score"] = round(min(1.0, 0.85 + 0.15*("SPINOUT" in m["kinds"])), 2)
    return merged


# ----------------------------- pass 2: render -----------------------------
def affected(active, byid, fi):
    """Vehicles that actually collided in this event, each from its own moment of contact.
    No 2D-overlap chaining: in an oblique view a far car and a near car can share image
    space without touching, which would wrongly inflate the wreck box."""
    aff = set()
    for e in active:
        inv = e.get("involved", {})
        ids = {i for i in e["ids"] if i in byid and inv.get(i, e["impact"]) <= fi}
        e["_aff"] = ids; aff |= ids
    return aff


def render(frame, perframe, active, fi, fps, name, n_total):
    vis = frame.copy(); H, W = vis.shape[:2]
    byid = {tid: (bb, stp, c) for tid, c, bb, stp, sm in perframe}
    n_now = sum(1 for _, c, *_ in perframe if c in VEHICLE)
    aff_all = affected(active, byid, fi)

    # EVERY vehicle gets its own box: green = normal, amber = stopped, red = in the crash
    for tid, c, bb, stp, sm in perframe:
        x1, y1, x2, y2 = map(int, bb)
        if tid in aff_all:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif c == PERSON:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 0), 2)
            cv2.putText(vis, "person", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1, cv2.LINE_AA)
        elif stp >= STOP_HOLD:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 170, 255), 2)
            cv2.putText(vis, "stopped", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 170, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # the accident zone: one bounding red box around the wreck, "ACCIDENT DETECTED" on top
    thick = 3 + int(1 + 1.5*max(0.0, math.sin(fi*0.7)))
    if aff_all:
        xs = [byid[i][0] for i in aff_all]
        x1 = int(min(b[0] for b in xs)) - 6; y1 = int(min(b[1] for b in xs)) - 6
        x2 = int(max(b[2] for b in xs)) + 6; y2 = int(max(b[3] for b in xs)) + 6
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), thick)
        label = "ACCIDENT DETECTED"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ly = (y1 - 8) if (y1 - 8 - th) > 52 else (y2 + th + 14)      # above the box, or below if it hits the banner
        lx = min(max(0, x1), W - tw - 12)
        cv2.rectangle(vis, (lx, ly - th - 9), (lx + tw + 12, ly + 5), (0, 0, 255), -1)
        cv2.putText(vis, label, (lx + 6, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if any(0 <= fi - e["impact"] <= 8 for e in active):        # brief flash at the instant of impact
        cv2.rectangle(vis, (0, 0), (W-1, H-1), (0, 0, 255), 20)

    # ONE clean top bar holds everything (no stacked labels)
    if active:
        lead = min(active, key=lambda e: e["impact"])
        cv2.rectangle(vis, (0, 0), (W, 46), (0, 0, 210), -1)
        cv2.putText(vis, f"{reason_of(lead['kinds']).upper()}      t={lead['first_t']}s      "
                         f"{max(len(aff_all),1)} vehicles involved      confidence {lead['score']:.2f}",
                    (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(vis, (0, 0), (452, 32), (35, 35, 35), -1)
        cv2.putText(vis, "SMART ACCIDENT DETECTION", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 0), 2, cv2.LINE_AA)

    # one clean status line at the bottom (count + honesty note)
    cv2.rectangle(vis, (0, H-30), (W, H), (0, 0, 0), -1)
    cv2.putText(vis, f"tracking {n_now} vehicles ({n_total} seen)   t={fi/fps:4.1f}s      "
                     f"Automated flag for human review  |  vision cues, not ground truth",
                (14, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return vis


# ----------------------------- driver -----------------------------
def process(video, out_dir):
    name = os.path.splitext(os.path.basename(video))[0]
    print(f"\n=== {name} ===")
    data = collect(video)
    events = confirm_events(data, _fps(video))
    fps = _fps(video)
    for e in events:
        print(f"  [ACCIDENT] t={e['first_t']:4}s  {reason_of(e['kinds'])}  conf={e['score']}  ids={sorted(e['ids'])}")

    cap = cv2.VideoCapture(video); W = int(cap.get(3)); H = int(cap.get(4))
    tmp = os.path.join(out_dir, f"{name}__tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fi, perframe in enumerate(data["fdata"]):
        ret, frame = cap.read()
        if not ret: break
        active = [e for e in events if fi >= e["impact"]]
        vw.write(render(frame, perframe, active, fi, fps, name, len(data["seen_ids"])))
    cap.release(); vw.release()

    final = os.path.join(out_dir, f"{name}_accident.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", final], check=True)
    os.remove(tmp)
    out = [{"type": "ACCIDENT", "reason": reason_of(e["kinds"]), "time_s": e["first_t"],
            "frame": int(e["impact"]), "ids": sorted(int(i) for i in e["ids"]),
            "bbox": [int(round(v)) for v in e["bbox"]], "confidence": e["score"]} for e in events]
    json.dump(out, open(os.path.join(out_dir, f"{name}_events.json"), "w"), indent=2)
    print(f"  -> {final}  ({len(events)} accident event(s))")
    return name, out


def _fps(video):
    cap = cv2.VideoCapture(video); f = cap.get(cv2.CAP_PROP_FPS) or 24.0; cap.release(); return f


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    proj = os.path.dirname(ROOT)
    default = [os.path.join(proj, f) for f in
               ["Car Accident 1.0.mov", "Car Accident 2.mov", "Car Accident 3.mov"]]
    vids = sys.argv[1:] or default
    summary = []
    for v in vids:
        if not os.path.exists(v): print(f"!! missing {v}"); continue
        summary.append(process(v, OUT_DIR))
    print("\n================ SUMMARY ================")
    for name, ev in summary:
        if ev:
            for e in ev: print(f"{name:20s} t={e['time_s']:>5}s  {e['reason']:<34s} conf={e['confidence']}")
        else: print(f"{name:20s} no accident flagged")
