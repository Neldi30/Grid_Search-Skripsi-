"""
record_data.py
Perekam data EAR + head pose via webcam laptop untuk grid search α, β, γ.
Replikasi logika EAR + solvePnP dari TestingActivity.kt (Android app).

Cara pakai:
    python record_data.py --subject taufik_01
    python record_data.py --subject teman_a --cam 1   # webcam ke-2

Output: recordings/<subject>.csv
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Konstanta — identik dengan TestingActivity.kt ────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
HEAD_LANDMARKS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    [   0.0,    0.0,    0.0],   # nose tip
    [   0.0, -330.0,  -65.0],   # chin
    [-225.0,  170.0, -135.0],   # left eye outer
    [ 225.0,  170.0, -135.0],   # right eye outer
    [-150.0, -150.0, -125.0],   # left mouth
    [ 150.0, -150.0, -125.0],   # right mouth
], dtype=np.float64)

T_BASE_DISPLAY = 0.24
ALPHA_DISPLAY  = 0.0015
BETA_DISPLAY   = 0.001
GAMMA_DISPLAY  = 0.0005

# ── Protokol perekaman ───────────────────────────────────────────────────
# Format: (start_s, duration_s, label, orientation, eye_state, color_bgr)
# eye_state: 0=terbuka, 1=tertutup, -1=tidak direkam (SIAP/JEDA/SELESAI)
PROTOCOL = [
    (  0.0,  5.0, "SIAP",                    "none",    -1, (128, 128, 128)),
    (  5.0, 20.0, "LURUS + MATA TERBUKA",    "frontal",  0, (94, 197, 34)),
    ( 25.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    ( 28.0, 15.0, "LURUS + MATA TERTUTUP",   "frontal",  1, (68, 68, 239)),
    ( 43.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    ( 46.0, 15.0, "TENGOK KANAN + TERBUKA",  "right",    0, (212, 182, 6)),
    ( 61.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    ( 64.0, 15.0, "TENGOK KANAN + TERTUTUP", "right",    1, (68, 68, 239)),
    ( 79.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    ( 82.0, 15.0, "TENGOK KIRI + TERBUKA",   "left",     0, (212, 182, 6)),
    ( 97.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    (100.0, 15.0, "TENGOK KIRI + TERTUTUP",  "left",     1, (68, 68, 239)),
    (115.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    (118.0, 15.0, "TUNDUK + TERBUKA",        "down",     0, (212, 182, 6)),
    (133.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    (136.0, 15.0, "TUNDUK + TERTUTUP",       "down",     1, (68, 68, 239)),
    (151.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    (154.0, 15.0, "TENGADAH + TERBUKA",      "up",       0, (212, 182, 6)),
    (169.0,  3.0, "JEDA",                    "none",    -1, (184, 163, 148)),
    (172.0, 15.0, "TENGADAH + TERTUTUP",     "up",       1, (68, 68, 239)),
    (187.0,  3.0, "SELESAI",                 "none",    -1, (94, 197, 34)),
]
TOTAL_DURATION = PROTOCOL[-1][0] + PROTOCOL[-1][1]
CALIB_PHASE_IDX = 1     # phase "LURUS + MATA TERBUKA"
CALIB_FRAME_COUNT = 30
PHASE_SETTLE_S = 1.0    # skip 1 detik awal tiap fase (transisi pose)


# ── Helper ───────────────────────────────────────────────────────────────
def wrap_angle(deg):
    a = deg % 360.0
    if a > 180.0:  a -= 360.0
    if a < -180.0: a += 360.0
    return a


def calculate_ear(eye_pts):
    """EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)  — identik Kotlin."""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C) if C > 1e-9 else 0.0


def calculate_head_pose(landmarks_xy, w, h, prev_rvec, prev_tvec):
    """Head pose via solvePnP (EPnP cold-start, ITERATIVE warm-start)."""
    image_points = np.array(
        [[landmarks_xy[i][0] * w, landmarks_xy[i][1] * h] for i in HEAD_LANDMARKS],
        dtype=np.float64,
    )
    camera_matrix = np.array([
        [w,   0.0, w / 2.0],
        [0.0, w,   h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    if prev_rvec is not None and prev_tvec is not None:
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            prev_rvec.copy(), prev_tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

    if not success:
        return 0.0, 0.0, 0.0, prev_rvec, prev_tvec

    rmat, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    yaw   = np.degrees(np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2)))
    roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    return wrap_angle(yaw), wrap_angle(pitch), wrap_angle(roll), rvec, tvec


def get_phase(t):
    for i, (start, dur, label, ori, eye, color) in enumerate(PROTOCOL):
        if start <= t < start + dur:
            return i, {
                "label": label, "orientation": ori, "eye_state": eye,
                "color": color, "phase_start": start, "phase_end": start + dur,
            }
    return -1, None


# ── Display ──────────────────────────────────────────────────────────────
def draw_overlay(frame, t, phase, ear, yaw, pitch, roll, t_dyn,
                 calibrating, n_calib, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (42, 23, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if phase is not None:
        label, color = phase["label"], phase["color"]
        rem = max(0.0, phase["phase_end"] - t)
        cv2.putText(frame, label, (20, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        cv2.putText(frame, f"{rem:4.1f}s", (w - 140, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (240, 232, 226), 2)

    pct = min(1.0, t / TOTAL_DURATION)
    cv2.rectangle(frame, (0, 88), (int(w * pct), 95), (212, 182, 6), -1)

    if calibrating:
        cv2.putText(frame, f"KALIBRASI {n_calib}/{CALIB_FRAME_COUNT}",
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (21, 204, 250), 2)

    info1 = f"EAR={ear:.3f}  Tdyn={t_dyn:.3f}"
    info2 = f"yaw={yaw:+6.1f}  pitch={pitch:+6.1f}  roll={roll:+6.1f}"
    info3 = f"t={t:6.1f}/{TOTAL_DURATION:.0f}s  FPS={fps:4.1f}"
    cv2.putText(frame, info1, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 232, 226), 1)
    cv2.putText(frame, info2, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (184, 163, 148), 1)
    cv2.putText(frame, info3, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (212, 234, 94), 1)


def show_idle_screen(cap, msg_lines):
    """Tampilkan frame webcam dengan instruksi sebelum mulai."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 80), (w, h // 2 + 80), (40, 30, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        for i, line in enumerate(msg_lines):
            y = h // 2 - 30 + i * 35
            cv2.putText(frame, line, (40, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (212, 182, 6), 2)
        cv2.imshow("Grid Search Recorder", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:   return True   # SPACE
        if key == 27:   return False  # ESC


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Nama subjek (mis. taufik_01)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--output-dir", default="recordings", help="Folder output CSV+MP4")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip menyimpan video .mp4 (default: simpan video)")
    parser.add_argument("--video-fps", type=float, default=30.0,
                        help="Target FPS untuk video file (default 30)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv   = out_dir / f"{args.subject}.csv"
    out_video = out_dir / f"{args.subject}.mp4"
    if out_csv.exists() or (not args.no_video and out_video.exists()):
        existing = []
        if out_csv.exists():   existing.append(out_csv.name)
        if out_video.exists(): existing.append(out_video.name)
        ans = input(f"File {', '.join(existing)} sudah ada. Timpa? [y/N]: ").strip().lower()
        if ans != "y":
            print("Dibatalkan.")
            return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa buka webcam index {args.cam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Probe ukuran frame aktual (webcam mungkin tidak honor set request)
    ok_probe, probe = cap.read()
    if not ok_probe:
        print("[ERROR] Tidak bisa baca frame dari webcam")
        sys.exit(1)
    actual_h, actual_w = probe.shape[:2]
    print(f"  webcam resolution aktual: {actual_w}x{actual_h}")

    # Setup video writer (kalau diaktifkan)
    video_writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(out_video), fourcc, args.video_fps, (actual_w, actual_h)
        )
        if not video_writer.isOpened():
            print(f"[WARN] Tidak bisa buat video writer di {out_video}. "
                  f"Lanjut tanpa save video.")
            video_writer = None
        else:
            print(f"  video akan disimpan ke: {out_video}")

    print("=" * 60)
    print(f"Subject  : {args.subject}")
    print(f"Output   : {out_csv}")
    print(f"Durasi   : {TOTAL_DURATION:.0f} detik (~3 menit)")
    print("=" * 60)

    started = show_idle_screen(cap, [
        "Tekan SPACE untuk mulai recording",
        "ESC untuk batal",
        "Pastikan wajah terlihat & cahaya cukup",
    ])
    if not started:
        cap.release(); cv2.destroyAllWindows()
        print("Dibatalkan.")
        return

    # ── Recording loop ──────────────────────────────────────────────────
    records = []
    fps_buf = []
    prev_rvec = prev_tvec = None
    yaw_off = pitch_off = roll_off = 0.0
    calib_collected = []
    calibrated = False

    t_start = time.time()
    print("[REC] Recording dimulai...\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Gagal baca frame")
            break
        h, w = frame.shape[:2]
        t_now = time.time() - t_start
        if t_now >= TOTAL_DURATION:
            break

        # Simpan frame ORIGINAL (non-mirror) ke video file untuk backup/re-extract
        if video_writer is not None:
            video_writer.write(frame)

        # Display mirror, process original
        frame_display = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS rolling (30 sample)
        fps_buf.append(time.time())
        if len(fps_buf) > 30: fps_buf.pop(0)
        fps = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0]) if len(fps_buf) > 1 else 0.0

        phase_idx, phase = get_phase(t_now)
        results = face_mesh.process(rgb)

        ear = 0.0
        yaw = pitch = roll = 0.0
        t_dyn = T_BASE_DISPLAY

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks_xy = np.array([[p.x, p.y] for p in lm], dtype=np.float64)

            ear = (calculate_ear(landmarks_xy[LEFT_EYE]) +
                   calculate_ear(landmarks_xy[RIGHT_EYE])) / 2.0

            yaw_r, pitch_r, roll_r, prev_rvec, prev_tvec = calculate_head_pose(
                landmarks_xy, w, h, prev_rvec, prev_tvec
            )

            # Kalibrasi: kumpulkan 30 frame setelah PHASE_SETTLE_S
            if (phase_idx == CALIB_PHASE_IDX and not calibrated and
                phase is not None and t_now - phase["phase_start"] >= PHASE_SETTLE_S):
                calib_collected.append((yaw_r, pitch_r, roll_r))
                if len(calib_collected) >= CALIB_FRAME_COUNT:
                    arr = np.array(calib_collected)
                    yaw_off, pitch_off, roll_off = arr.mean(axis=0)
                    calibrated = True
                    print(f"[CALIB] yaw_off={yaw_off:+.2f}  "
                          f"pitch_off={pitch_off:+.2f}  roll_off={roll_off:+.2f}")

            yaw   = wrap_angle(yaw_r   - yaw_off)
            pitch = wrap_angle(pitch_r - pitch_off)
            roll  = wrap_angle(roll_r  - roll_off)
            t_dyn = T_BASE_DISPLAY - (ALPHA_DISPLAY * abs(yaw) +
                                     BETA_DISPLAY  * abs(pitch) +
                                     GAMMA_DISPLAY * abs(roll))

            # Rekam frame jika fase memiliki ground truth eye_state (>= 0)
            # & sudah lewat masa transisi PHASE_SETTLE_S detik
            if (phase is not None and phase["eye_state"] >= 0 and
                t_now - phase["phase_start"] >= PHASE_SETTLE_S):
                records.append({
                    "frame_id":    len(records),
                    "t_s":         round(t_now, 3),
                    "ear":         round(ear, 5),
                    "yaw":         round(yaw, 3),
                    "pitch":       round(pitch, 3),
                    "roll":        round(roll, 3),
                    "orientation": phase["orientation"],
                    "eye_state":   phase["eye_state"],
                    "phase":       phase["label"],
                })

        draw_overlay(
            frame_display, t_now, phase, ear, yaw, pitch, roll, t_dyn,
            calibrating=(phase_idx == CALIB_PHASE_IDX and not calibrated),
            n_calib=len(calib_collected), fps=fps,
        )
        cv2.imshow("Grid Search Recorder", frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[ABORT] Dihentikan user (ESC).")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    if video_writer is not None:
        video_writer.release()

    if not records:
        print("[ERROR] Tidak ada data terekam — wajah tidak terdeteksi?")
        return

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print(f"Recording selesai: {len(records)} frame")
    print(f"CSV tersimpan: {out_csv}")
    if video_writer is not None:
        print(f"Video tersimpan: {out_video}")
    print("=" * 60)
    print("Distribusi data per (orientation, eye_state):")
    dist = df.groupby(["orientation", "eye_state"]).size().rename("count")
    print(dist.to_string())
    print("\nRange head pose per orientation:")
    summary = df.groupby("orientation").agg({
        "yaw":   ["mean", "min", "max"],
        "pitch": ["mean", "min", "max"],
        "roll":  ["mean", "min", "max"],
    }).round(2)
    print(summary.to_string())
    print("\nNext: jalankan `python grid_search.py` untuk cari best α, β, γ.")


if __name__ == "__main__":
    main()
