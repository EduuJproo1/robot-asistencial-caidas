"""
Script de validación interactiva mediante Webcam (v2 con Geometría 3D).
"""

import argparse
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
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def draw_hud(frame: np.ndarray, result, fps: float, confirm_sec: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 72), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if result.state == FallState.CAIDA_CONFIRMADA:
        status_color = (0, 0, 255)
        status_txt = "ESTADO: CAIDA CONFIRMADA"
    elif result.state == FallState.CONFIRMANDO:
        status_color = (0, 165, 255)
        status_txt = f"ESTADO: EVALUANDO SUELO ({result.time_in_fall:.1f}s / {confirm_sec:.1f}s)"
    else:
        status_color = (50, 220, 100)
        status_txt = "ESTADO: NORMAL"

    # Fila 1: Estado principal y FPS
    cv2.putText(frame, status_txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (w - 115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Fila 2: Métricas 3D
    postura_txt = "EN SUELO / HORIZONTAL" if result.is_lying_or_grounded else "VERTICAL / ERGUIDO"
    postura_color = (0, 165, 255) if result.is_lying_or_grounded else (180, 220, 180)

    info_txt = f"TFLite: {result.fall_confidence * 100:4.1f}% | Torso 3D: {result.torso_angle_3d:4.1f} deg | Aspect: {result.aspect_ratio:3.2f} | "
    cv2.putText(frame, info_txt, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    (t_w, _), _ = cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, f"Postura: {postura_txt}", (15 + t_w, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, postura_color, 1, cv2.LINE_AA)

    # Banner inferior de alarma confirmada
    if result.state == FallState.CAIDA_CONFIRMADA:
        banner_h = 45
        banner_y = h - banner_h - 10
        cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h), (0, 0, 220), -1)
        cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h), (255, 255, 255), 2)
        cv2.putText(frame, "ALERTA CRITICA: PERSONA TENDIDA EN EL SUELO", (25, banner_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def run_webcam(camera_index: int, conf_thresh: float, confirm_sec: float, mirror: bool):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir la cámara con índice {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FallDetector(
        model_path="models/fall_detection_transformer.tflite",
        fall_confidence_threshold=conf_thresh,
        confirmation_time_sec=confirm_sec
    )

    print("\n" + "=" * 65)
    print("📹 Monitor de Detección de Caídas v2 (Geometría 3D + Suelo)")
    print(f"⏱️ Tiempo de confirmación en suelo: {confirm_sec}s")
    print(f"🎯 Umbral de disparo TFLite: {conf_thresh * 100:.0f}%")
    print("Controles: 'R' = Resetear FSM | 'Q' o 'ESC' = Salir")
    print("=" * 65 + "\n")

    prev_time = time.monotonic()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        now = time.monotonic()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / max(dt, 1e-4)

        result = detector.process_frame(frame, current_timestamp=now)

        frame_pose = draw_punpayut_landmarks(frame, result.pose_landmarks)
        display_frame = draw_hud(frame_pose, result, fps, confirm_sec)

        cv2.imshow("Robot Asistencial - Monitor Webcam v2", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            detector.reset()
            print("🔄 FSM reseteada manualmente a NORMAL.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoreo de caídas en vivo por Webcam v2")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara")
    parser.add_argument("--conf", type=float, default=0.80, help="Umbral TFLite (default: 0.80)")
    parser.add_argument("--confirm", type=float, default=1.2, help="Segundos de confirmación (default: 1.2s)")
    parser.add_argument("--no-mirror", action="store_true", help="Desactivar efecto espejo")
    args = parser.parse_args()

    run_webcam(
        camera_index=args.camera,
        conf_thresh=args.conf,
        confirm_sec=args.confirm,
        mirror=not args.no_mirror
    )