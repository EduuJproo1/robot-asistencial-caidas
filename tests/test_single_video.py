"""
Script de inspección visual unitario con telemetría geométrica calibrada.
Sincronizado con FallDetector v3.
"""

import argparse
from pathlib import Path
import time
import cv2
import mediapipe as mp
import numpy as np

from src.fall_detector import FallDetector, FallState

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def draw_punpayut_landmarks(frame: np.ndarray, landmarks) -> np.ndarray:
    if not landmarks:
        return frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_drawing.draw_landmarks(
        frame_rgb,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    return cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)


def draw_hud(frame: np.ndarray, result, fps: float, confirm_sec: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if result.state == FallState.CAIDA_CONFIRMADA:
        status_color = (0, 0, 255)
        status_txt = "ESTADO: CAIDA CONFIRMADA"
    elif result.state == FallState.CONFIRMANDO:
        status_color = (0, 165, 255)
        status_txt = f"ESTADO: CONFIRMANDO POSTURA ({result.time_in_fall:.1f}s / {confirm_sec:.1f}s)"
    else:
        status_color = (50, 220, 100)
        status_txt = "ESTADO: NORMAL"

    cv2.putText(frame, status_txt, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    postura_txt = "SUELO (HORIZONTAL)" if result.is_grounded else "VERTICAL / ERGUIDO"
    postura_color = (0, 165, 255) if result.is_grounded else (180, 220, 180)

    info_txt = f"TFLite: {result.fall_confidence * 100:4.1f}% | Torso: {result.torso_angle:4.1f} deg | Aspect: {result.aspect_ratio:3.2f} | "
    cv2.putText(frame, info_txt, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    (t_w, _), _ = cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, f"Postura: {postura_txt}", (15 + t_w, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, postura_color, 1, cv2.LINE_AA)

    if result.state == FallState.CAIDA_CONFIRMADA:
        banner_h = 45
        banner_y = h - banner_h - 10
        cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h), (0, 0, 220), -1)
        cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h), (255, 255, 255), 2)
        cv2.putText(frame, "ALERTA: CAIDA CRITICA DETECTADA EN SUELO", (25, banner_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def run_test(video_path: str, conf_thresh: float, confirm_sec: float):
    video_file = Path(video_path).resolve()
    if not video_file.exists():
        print(f"❌ Error: Video no encontrado en: {video_file}")
        return

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"❌ Error al abrir el video: {video_file}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps) or video_fps > 120:
        video_fps = 30.0
    dt = 1.0 / video_fps

    # Inicializar con los umbrales estadísticos óptimos
    detector = FallDetector(
        model_path="models/fall_detection_transformer.tflite",
        fall_confidence_threshold=conf_thresh,
        confirmation_time_sec=confirm_sec,
        torso_angle_threshold=40.0,
        aspect_ratio_threshold=0.85
    )

    modo_txt = "Directo Punpayut" if confirm_sec <= 0.0 else f"Robot FSM Hibrido ({confirm_sec}s)"
    print("\n" + "=" * 65)
    print(f"🎬 Video: {video_file.name}")
    print(f"⚙️ Modo:  {modo_txt}")
    print(f"🎯 Umbral Probabilidad: {conf_thresh * 100:.0f}%")
    print("=" * 65 + "\n")

    simulated_timestamp = 0.0
    paused = False
    max_conf = 0.0
    fall_triggered = False

    cv2.namedWindow("Fall Detection - Inspeccion Unitaria", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Detection - Inspeccion Unitaria", 960, 540)

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t_start = time.perf_counter()
            simulated_timestamp += dt

            result = detector.process_frame(frame, current_timestamp=simulated_timestamp)

            if result.fall_confidence > max_conf:
                max_conf = result.fall_confidence
            if result.state == FallState.CAIDA_CONFIRMADA:
                fall_triggered = True

            t_elapsed = time.perf_counter() - t_start
            current_fps = 1.0 / max(t_elapsed, 1e-4)

            frame_with_pose = draw_punpayut_landmarks(frame, result.pose_landmarks)
            final_frame = draw_hud(frame_with_pose, result, current_fps, confirm_sec)

            cv2.imshow("Fall Detection - Inspeccion Unitaria", final_frame)

        key = cv2.waitKey(int(dt * 1000)) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    print(f"📊 Resumen del video:")
    print(f"   - Confianza Máxima TFLite: {max_conf * 100:.2f}%")
    print(f"   - Caída Confirmada: {'SI (ALERTA)' if fall_triggered else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador de detección de caídas unitario")
    parser.add_argument("--video", type=str, required=True, help="Ruta al video .mp4")
    parser.add_argument("--conf", type=float, default=0.85, help="Umbral de confianza (default: 0.85)")
    parser.add_argument("--confirm", type=float, default=0.8, help="Segundos de permanencia requeridos")
    args = parser.parse_args()

    run_test(video_path=args.video, conf_thresh=args.conf, confirm_sec=args.confirm)