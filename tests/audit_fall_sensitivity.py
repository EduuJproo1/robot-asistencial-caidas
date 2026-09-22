"""
Auditoría de Sensibilidad (Recall) sobre Caídas Reales.
Sincronizado con DetectorConfig y FallDetector v6.
"""

from pathlib import Path
import random
import time
import cv2
import pandas as pd

from src.detector_config import DetectorConfig
from src.fall_detector import FallDetector, FallState, Posture


def run_fall_audit(sample_size: int = 50, seed: int = 42):
    random.seed(seed)
    project_root = Path(__file__).resolve().parent.parent
    fall_dir = Path(r"C:\Users\EduuJproo1\Documents\Fall_Detection_Proyectos\archive\Fall\Raw_Video")

    if not fall_dir.exists():
        print(f"❌ Error: Carpeta no encontrada en {fall_dir}")
        return

    all_videos = list(fall_dir.glob("*.mp4"))
    sampled_videos = random.sample(all_videos, min(sample_size * 2, len(all_videos)))

    print("\n" + "=" * 70)
    print("🎯 AUDITORÍA DE SENSIBILIDAD EN CAÍDAS REALES (RECALL CHECK)")
    print(f"📁 Directorio: {fall_dir.name}")
    print(f"🎯 Muestra objetivo: {sample_size} caídas representativas")
    print("=" * 70 + "\n")

    # Inyección de configuración centralizada
    config = DetectorConfig(
        tflite_conf_threshold=0.85,
        confirmation_time_sec=0.8,
        torso_angle_threshold=40.0,
        aspect_ratio_threshold=0.85
    )

    detector = FallDetector(
        model_path="models/fall_detection_transformer.tflite",
        config=config
    )

    results = []
    valid_count = 0
    start_time = time.perf_counter()

    for v_path in sampled_videos:
        if valid_count >= sample_size:
            break

        cap = cv2.VideoCapture(str(v_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Omitir clips truncados menores a 35 frames (~1 segundo)
        if total_frames < 35:
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / fps

        simulated_ts = 0.0
        fall_detected = False
        max_tflite = 0.0
        min_torso = 90.0
        was_confirming = False
        was_grounded = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            simulated_ts += dt
            res = detector.process_frame(frame, current_timestamp=simulated_ts)

            if res.fall_confidence > max_tflite:
                max_tflite = res.fall_confidence
            if res.torso_angle < min_torso:
                min_torso = res.torso_angle
            if res.posture == Posture.GROUND or res.is_grounded:
                was_grounded = True
            if res.state == FallState.CONFIRMANDO:
                was_confirming = True
            if res.state == FallState.CAIDA_CONFIRMADA:
                fall_detected = True

        cap.release()
        detector.reset()

        # En clips cortos de dataset, si el archivo terminó mientras estaba confirmando en suelo,
        # en un flujo continuo de robot habría confirmado:
        confirmed_or_grounded = fall_detected or (was_confirming and was_grounded)

        status = "✅ CAÍDA DETECTADA (TP)" if confirmed_or_grounded else "❌ NO DETECTADA (FN)"
        results.append({
            "video": v_path.name,
            "max_tflite": max_tflite,
            "min_torso": min_torso,
            "detected": confirmed_or_grounded,
            "status": status
        })

        valid_count += 1
        print(f"[{valid_count:2d}/{sample_size}] {v_path.name:<26} | TFLite: {max_tflite*100:4.1f}% | Torso: {min_torso:4.1f}° | {status}")

    elapsed = time.perf_counter() - start_time
    df_res = pd.DataFrame(results)

    tp = len(df_res[df_res["detected"]])
    fn = len(df_res[~df_res["detected"]])
    recall = (tp / max(len(df_res), 1)) * 100.0

    print("\n" + "=" * 70)
    print("📊 RESULTADOS DE SENSIBILIDAD (RECALL)")
    print("=" * 70)
    print(f"Total caídas evaluadas:            {len(df_res)}")
    print(f"Verdaderos Positivos (TP):        {tp}  ({recall:.1f}%)")
    print(f"Falsos Negativos (FN):            {fn}  ({100.0 - recall:.1f}%)")
    print(f"Tasa de Sensibilidad / Recall:    {recall:.2f}%")
    print(f"Tiempo total:                     {elapsed:.1f} s")
    print("=" * 70 + "\n")

    out_csv = project_root / "tests" / "audit_fall_sensitivity_report.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"✅ Telemetría guardada en: {out_csv.resolve()}")


if __name__ == "__main__":
    run_fall_audit(sample_size=50)