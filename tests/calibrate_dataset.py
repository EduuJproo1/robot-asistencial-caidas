"""
Script de calibración estadística representativa para tesis de título.
Aplica muestreo pseudoaleatorio estratificado y filtrado de videos truncados.
"""

import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src.fall_detector import FallDetector


def calibrate_dataset(sample_size: int = 30, min_frames: int = 45, seed: int = 42):
    random.seed(seed)
    
    fall_dir = Path(r"C:\Users\EduuJproo1\Documents\Fall_Detection_Proyectos\archive\Fall\Raw_Video")
    nofall_dir = Path(r"C:\Users\EduuJproo1\Documents\Fall_Detection_Proyectos\archive\No_Fall\Raw_Video")

    detector = FallDetector(confirmation_time_sec=0.0)

    results = []

    def process_folder(folder_path: Path, label: str):
        all_videos = list(folder_path.glob("*.mp4"))
        if not all_videos:
            print(f"❌ No se encontraron videos en {folder_path}")
            return

        # Muestreo aleatorio en lugar de tomar los primeros en orden
        sampled_videos = random.sample(all_videos, min(sample_size * 2, len(all_videos)))
        valid_count = 0

        print(f"\nAnalizando muestra representativa de '{label}'...")

        for v_path in sampled_videos:
            if valid_count >= sample_size:
                break

            cap = cv2.VideoCapture(str(v_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Filtro de duración: descartar clips truncados
            if total_frames < min_frames:
                cap.release()
                continue

            max_conf = 0.0
            peak_frame_idx = 0
            frame_records = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                res = detector.process_frame(frame)

                if res.fall_confidence > max_conf:
                    max_conf = res.fall_confidence
                    peak_frame_idx = frame_idx

                if res.person_detected:
                    frame_records.append({
                        "frame": frame_idx,
                        "torso_angle": res.torso_angle,
                        "aspect_ratio": res.aspect_ratio
                    })
                frame_idx += 1

            cap.release()
            detector.reset()

            if not frame_records:
                continue

            df_frames = pd.DataFrame(frame_records)

            # Para caídas: analizar la fase POST-IMPACTO (después del pico de la red)
            # Para no-caídas: analizar el último tercio del video (postura final sostenida)
            if label == "Fall":
                post_impact = df_frames[df_frames["frame"] >= peak_frame_idx]
                eval_slice = post_impact.tail(15) if len(post_impact) >= 5 else df_frames.tail(10)
            else:
                eval_slice = df_frames.tail(15)

            final_angle = float(eval_slice["torso_angle"].median())
            final_ratio = float(eval_slice["aspect_ratio"].median())

            results.append({
                "video": v_path.name,
                "label": label,
                "total_frames": total_frames,
                "max_tflite_conf": max_conf,
                "final_torso_angle": final_angle,
                "final_aspect_ratio": final_ratio
            })
            valid_count += 1
            print(f"  [{valid_count}/{sample_size}] {v_path.name} | Conf Max: {max_conf*100:4.1f}% | Torso: {final_angle:4.1f}° | Aspect: {final_ratio:3.2f}")

    process_folder(fall_dir, "Fall")
    process_folder(nofall_dir, "No_Fall")

    df_res = pd.DataFrame(results).dropna()

    print("\n" + "=" * 70)
    print("📊 DISTRIBUCIONES ESTADÍSTICAS DEL DATASET (MEDIANA, MEDIA Y DESV. EST.)")
    print("=" * 70)

    stats = df_res.groupby("label")[["max_tflite_conf", "final_torso_angle", "final_aspect_ratio"]].agg(["mean", "std", "median"])
    print(stats.to_string())

    print("\n" + "-" * 70)
    print("🎯 PUNTOS DE SEPARACIÓN MATEMÁTICOS ÓPTIMOS:")
    
    fall_ang_med = df_res[df_res["label"] == "Fall"]["final_torso_angle"].median()
    nofall_ang_med = df_res[df_res["label"] == "No_Fall"]["final_torso_angle"].median()
    opt_angle = (fall_ang_med + nofall_ang_med) / 2.0

    fall_ratio_med = df_res[df_res["label"] == "Fall"]["final_aspect_ratio"].median()
    nofall_ratio_med = df_res[df_res["label"] == "No_Fall"]["final_aspect_ratio"].median()
    opt_ratio = (fall_ratio_med + nofall_ratio_med) / 2.0

    print(f"  • Torso en Suelo (Mediana Fall):        {fall_ang_med:4.1f}°")
    print(f"  • Torso Erguido/Sentado (Mediana No-Fall): {nofall_ang_med:4.1f}°")
    print(f"  --> Umbral Óptimo de Torso:             {opt_angle:4.1f}°")
    print(f"  • Aspect Ratio en Suelo (Mediana Fall):   {fall_ratio_med:4.2f}")
    print(f"  • Aspect Ratio Erguido (Mediana No-Fall): {nofall_ratio_med:4.2f}")
    print(f"  --> Umbral Óptimo de Aspect Ratio:      {opt_ratio:4.2f}")
    print("-" * 70)

    # Guardar resultados en CSV para adjuntar al informe o graficar
    output_csv = Path("calibration_dataset_metrics.csv")
    df_res.to_csv(output_csv, index=False)
    print(f"✅ Telemetría guardada en: {output_csv.resolve()}\n")


if __name__ == "__main__":
    calibrate_dataset(sample_size=30, min_frames=45, seed=42)