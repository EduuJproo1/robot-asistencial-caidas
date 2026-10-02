"""
Punto de Entrada Principal (Main Pipeline) del Sistema de Detección y Guiado.
Integra Visión (FallDetector), Alertas (AlertManager) y Voz (VoiceManager).
"""

import argparse
import os
from pathlib import Path
import time
import cv2
from dotenv import load_dotenv
import mediapipe as mp
import numpy as np

from src.fall_detector import FallDetector, FallState, Posture
from src.alert_manager import AlertManager
from src.detector_config import DetectorConfig
from src.voice_manager import VoiceManager, VoiceCommand

load_dotenv()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

_POSTURE_LABELS = {
    Posture.STANDING: "VERTICAL / ERGUIDO",
    Posture.GROUND: "SUELO (HORIZONTAL)",
    Posture.SEATED_STABLE: "SENTADO (ESTABLE)",
    Posture.AMBIGUOUS: "POSTURA AMBIGUA",
}

_POSTURE_COLORS = {
    Posture.STANDING: (180, 220, 180),
    Posture.GROUND: (0, 165, 255),
    Posture.SEATED_STABLE: (255, 220, 130),
    Posture.AMBIGUOUS: (150, 150, 150),
}


def draw_punpayut_landmarks(frame: np.ndarray, landmarks) -> np.ndarray:
    """Dibuja sobre el frame los landmarks y conexiones estimados por MediaPipe Pose."""
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


def draw_hud_and_telemetry(frame: np.ndarray, result, fps: float, confirm_sec: float) -> np.ndarray:
    """
    Superpone sobre el frame el estado del detector y la telemetría visual.

    Muestra el estado de la FSM, confianza TFLite, postura, métricas geométricas
    y, cuando está disponible, la orientación y distancia estimadas del objetivo.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel superior translúcido (72px)
    cv2.rectangle(overlay, (0, 0), (w, 72), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if result.state == FallState.CAIDA_CONFIRMADA:
        status_color = (0, 0, 255)
        status_txt = "ESTADO: CAIDA CONFIRMADA"
    elif result.state == FallState.CONFIRMANDO:
        status_color = (0, 165, 255)
        status_txt = f"ESTADO: CONFIRMANDO SUELO ({result.time_in_fall:.1f}s / {confirm_sec:.1f}s)"
    else:
        status_color = (50, 220, 100)
        status_txt = "ESTADO: NORMAL"

    cv2.putText(frame, status_txt, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    postura_txt = _POSTURE_LABELS.get(result.posture, "DESCONOCIDO")
    postura_color = _POSTURE_COLORS.get(result.posture, (200, 200, 200))

    info_txt = f"TFLite: {result.fall_confidence * 100:4.1f}% | Torso: {result.torso_angle:4.1f} deg | Aspect: {result.aspect_ratio:3.2f} | "
    cv2.putText(frame, info_txt, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    (t_w, _), _ = cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, f"Postura: {postura_txt}", (15 + t_w, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, postura_color, 1, cv2.LINE_AA)

    # --- Retícula de Navegación y Telemetría Robótica ---
    nav = result.telemetry
    if nav.is_target_valid:
        cx, cy = nav.target_center_px
        center_x = w // 2

        # Marcador central discreto (pequeña cruz en vez de línea completa)
        cv2.drawMarker(frame, (center_x, cy), (140, 140, 140), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

        # Vector de orientación hacia el objetivo (Cian)
        cv2.line(frame, (center_x, cy), (cx, cy), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2, cv2.LINE_AA)

        # Barra inferior
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 18, 22), -1)
        direction = "HORARIO (D)" if nav.bearing_deg > 0 else "ANTIHORARIO (I)"
        if abs(nav.bearing_deg) < 3.0:
            direction = "CENTRADO"

        cmd_txt = (
            f"BASE MOVIL -> Azimut: {nav.bearing_deg:+4.1f} deg [{direction}] | "
            f"Dist. est.: {nav.estimated_distance_m:4.2f} m"
        )
        cv2.putText(frame, cmd_txt, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)

    if result.state == FallState.CAIDA_CONFIRMADA:
        banner_h = 40
        banner_y = h - 85
        cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h), (0, 0, 220), -1)
        cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h), (255, 255, 255), 2)
        cv2.putText(frame, "ALERTA: NOTIFICACION DE AUXILIO DESPACHADA", (25, banner_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def run_pipeline(source):
    """
    Ejecuta el pipeline integrado de visión, voz y alertas.

    La fuente puede corresponder a una webcam o a un archivo de video.
    El ciclo principal procesa comandos de voz, ejecuta el detector y genera
    alertas cuando corresponde.
    """
    is_webcam = str(source).isdigit() or source == "0"
    cap = cv2.VideoCapture(int(source) if is_webcam else source)

    if not cap.isOpened():
        print(f"❌ Error al abrir la fuente: {source}")
        return

    config = DetectorConfig(
        tflite_conf_threshold=0.85,
        confirmation_time_sec=1.2,
        torso_angle_threshold=40.0,
        aspect_ratio_threshold=0.85
    )

    detector = FallDetector(
        model_path="models/fall_detection_transformer.tflite",
        config=config
    )

    alert_mgr = AlertManager(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        cooldown_sec=20.0,
        evidence_dir="evidences"
    )

    voice_mgr = VoiceManager(model_path="models/vosk-model-es")
    voice_mgr.start()

    print("\n" + "=" * 65)
    print("🚀 PIPELINE ROBÓTICO: Detección Visual + Alertas + Control por Voz")
    print(f"📹 Origen: {'Webcam (' + str(source) + ')' if is_webcam else Path(source).name}")
    print(f"🤖 Despacho Telegram: {'CONFIGURADO (.env)' if alert_mgr.bot_token else 'DESACTIVADO'}")
    print("🎙️ Interacción de Voz: INICIANDO EN SEGUNDO PLANO")
    print("Controles: 'Q' o 'ESC' para salir | 'R' para resetear")
    print("=" * 65 + "\n")

    prev_time = time.monotonic()

    try:
        result = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = time.monotonic()
            fps = 1.0 / max(now - prev_time, 1e-4)
            prev_time = now

            # 1. Monitoreo de comandos de voz no-bloqueante
            voice_cmd = voice_mgr.get_next_command()
            
            if voice_cmd == VoiceCommand.CANCELAR:
                print("🛑 [Interrupción por Voz] Comando de CANCELACIÓN recibido.")

                if (
                    result is not None
                    and result.state in (FallState.CONFIRMANDO, FallState.CAIDA_CONFIRMADA)
                ):
                    msg_cancel = (
                        "✅ *ALERTA CANCELADA*\n"
                        "El usuario ha confirmado por voz que se encuentra bien.\n"
                        "No se requiere asistencia."
                    )
                    alert_mgr._executor.submit(
                        alert_mgr._send_telegram_worker,
                        msg_cancel,
                        None
                    )

                detector.reset()
            
            elif voice_cmd == VoiceCommand.EMERGENCIA:
                print("🚨 [Interrupción por Voz] Comando de AYUDA recibido. Forzando alerta.")
                # Disparo inmediato sin importar la cámara
                alert_mgr.trigger_fall_alert(
                    frame_bgr=frame,
                    fall_confidence=1.0,
                    torso_angle=0.0,
                    time_in_fall=0.0,
                    emergency_by_voice=True
                )

            # 2. Procesamiento de Visión
            result = detector.process_frame(frame, current_timestamp=now)

            annotated = draw_punpayut_landmarks(frame.copy(), result.pose_landmarks)
            display_frame = draw_hud_and_telemetry(annotated, result, fps, detector.cfg.confirmation_time_sec)

            # 3. Disparo de Alerta Visual
            if result.state == FallState.CAIDA_CONFIRMADA:
                alert_mgr.trigger_fall_alert(
                    frame_bgr=display_frame,
                    fall_confidence=result.fall_confidence,
                    torso_angle=result.torso_angle,
                    time_in_fall=result.time_in_fall
                )

            cv2.imshow("Robot Asistencial - Pipeline Integrado", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                detector.reset()
                print("🔄 FSM reseteada a NORMAL desde teclado.")
    finally:
        print("\nApagando sistema de forma segura...")
        cap.release()
        cv2.destroyAllWindows()
        alert_mgr.shutdown()
        voice_mgr.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Principal Integrado")
    parser.add_argument("--source", type=str, default="0", help="0 para Webcam o ruta al video .mp4")
    args = parser.parse_args()

    run_pipeline(source=args.source)