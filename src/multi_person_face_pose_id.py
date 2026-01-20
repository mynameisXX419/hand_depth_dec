#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-model runnable demo:
YOLOv8 + RTMPose + InsightFace
Face ID (int) + Face–Skeleton binding

Run:
    python multi_person_face_pose_id.py --video your.mp4
"""

import cv2
import numpy as np
import argparse
import time

# ================= YOLO =================
from ultralytics import YOLO

# ================= RTMPose =================
from mmpose.apis import inference_topdown, init_model as init_pose_model
from mmdet.apis import init_detector

# ================= InsightFace =================
from insightface.app import FaceAnalysis

# ============================================================
# Utils
# ============================================================
def l2norm(x, eps=1e-12):
    return x / (np.linalg.norm(x) + eps)

def cosine_sim(a, b):
    return float(np.dot(l2norm(a), l2norm(b)))

def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6
    return inter / ua

# ============================================================
# Face embedding -> integer ID
# ============================================================
class FaceIDMapper:
    def __init__(self, sim_thresh=0.45):
        self.sim_thresh = sim_thresh
        self.db = {}
        self.next_id = 0

    def register(self, emb):
        pid = self.next_id
        self.db[pid] = l2norm(emb)
        self.next_id += 1
        return pid

    def match(self, emb):
        if not self.db:
            return None, 0.0
        emb = l2norm(emb)
        best_id, best_sim = None, -1
        for pid, ref in self.db.items():
            sim = cosine_sim(emb, ref)
            if sim > best_sim:
                best_id, best_sim = pid, sim
        if best_sim >= self.sim_thresh:
            conf = (best_sim - self.sim_thresh) / (1 - self.sim_thresh)
            return best_id, conf
        return None, 0.0

# ============================================================
# Face–Skeleton Binder
# ============================================================
class Binder:
    def __init__(self):
        self.tracks = []

    def bind(self, person_boxes, faces, face_ids):
        if len(self.tracks) != len(person_boxes):
            self.tracks = [{"pid": None, "conf": 0.0, "streak": 0}
                           for _ in person_boxes]

        face_boxes = [f.bbox.astype(int).tolist() for f in faces]

        for i, pbox in enumerate(person_boxes):
            best = None
            best_cost = 1e9
            for j, fbox in enumerate(face_boxes):
                iou = bbox_iou(pbox, fbox)
                dist = np.linalg.norm(np.array(bbox_center(pbox)) -
                                       np.array(bbox_center(fbox)))
                if iou < 0.02 and dist > 200:
                    continue
                cost = dist / 300 + (1 - iou)
                if cost < best_cost:
                    best_cost = cost
                    best = j

            track = self.tracks[i]
            track["conf"] *= 0.9

            if best is None:
                track["streak"] = max(0, track["streak"] - 1)
                continue

            pid, conf = face_ids[best]
            if pid is None:
                continue

            if track["pid"] == pid:
                track["streak"] += 1
                track["conf"] = max(track["conf"], conf)
            else:
                if track["streak"] < 3 or conf > track["conf"]:
                    track["pid"] = pid
                    track["conf"] = conf
                    track["streak"] = 1

        return self.tracks

# ============================================================
# Main
# ============================================================
def main(video_path):
    # -------- Models --------
    yolo = YOLO("yolov8n.pt")

    pose_cfg = "models/rtmpose/rtmpose-m_8xb256-420e_body8-256x192.py"
    pose_ckpt = "models/rtmpose/rtmpose-m_8xb256-420e_body8-256x192-6f1b3b82_20230504.pth"
    pose_model = init_pose_model(pose_cfg, pose_ckpt, device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu")

    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    id_mapper = FaceIDMapper()
    binder = Binder()

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- YOLO ----------
        results = yolo(frame, conf=0.5, classes=[0])[0]
        person_boxes = [list(map(int, b.xyxy[0].tolist())) for b in results.boxes]

        # ---------- RTMPose ----------
        persons = [{"bbox": box} for box in person_boxes]
        pose_results = inference_topdown(pose_model, frame, persons, bbox_format="xyxy")

        # ---------- Face ----------
        faces = face_app.get(frame)

        face_ids = []
        for f in faces:
            pid, conf = id_mapper.match(f.embedding)
            if pid is None:
                pid = id_mapper.register(f.embedding)
                conf = 1.0
            face_ids.append((pid, conf))

        tracks = binder.bind(person_boxes, faces, face_ids)

        # ---------- Draw ----------
        for i, box in enumerate(person_boxes):
            pid = tracks[i]["pid"]
            txt = f"ID:{pid}" if pid is not None else "ID:?"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            cv2.putText(frame, txt, (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Multi-Person ID Binding", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    main(args.video)
