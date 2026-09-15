"""
Auditoría focalizada sobre los Falsos Positivos históricos de Punpayut.
Evalúa la eficacia de la verificación geométrica y la FSM (RF3).
"""

from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd

from src.fall_detector import FallDetector, FallState


def run_fp_audit():
    # 1. Rutas de archivos
    project_root = Path(__file__).resolve().parent.parent
    csv_candidates = [
        project_root / "tests" / "metricas_punpayut_dataset.csv",
        project_root / "metricas_punpayut_dataset.csv"
    ]

    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if not csv_path:
        print(f"❌ Error: No se encontró 'metricas_punpayut_dataset.csv' en tests/ ni en la raíz.")
        return

    nofall_dir = Path(r"C:\Users\EduuJproo1\Documents\Fall_Detection_Proyectos\archive\No_Fall\Raw_Video")
    if not nofall_dir.exists():
        print(f"❌ Error: No se encontró la carpeta de videos No_Fall en: {nofall_dir}")
        return

    # 2. Cargar CSV y filtrar Falsos Positivos
    df_raw = pd.read_csv(csv_path)
    
    # Normalizar nombres de columnas (eliminar espacios en blanco)
    df_raw.columns = [col.strip() for col in df_raw.columns]

    # Detectar columna de video y condición de FP
    col_video = next((c for c in df_raw.columns if "video" in c.lower()), df_raw.columns[0])
    
    # Filtrar filas que contengan "Falso Positivo" o "FP" en cualquier columna de texto
    fp_mask = df_raw.astype(str).apply(lambda row: row.str.contains("Falso Positivo|FP", case=False).any(), axis=1)
    df_fps = df_raw[fp_mask].copy()

    total_fps_original = len(df_fps)
    print("\n" + "=" * 70)
    print(f"🔍 AUDITORÍA DE FALSOS POSITIVOS (RF3)")
    print(f"📁 Archivo de referencia: {csv_path.name}")
    print(f"🎯 Total de Falsos Positivos a auditar: {total_fps_original}")
    print("=" * 70 + "\n")

    if total_fps_original == 0:
        print("⚠️ No se encontraron registros con etiqueta 'Falso Positivo' o 'FP' en el CSV.")
        return

    # 3. Inicializar detector híbrido con parámetros calibrados
    detector = FallDetector(
        model_path="models/fall_detection_transformer.tflite",
        fall_confidence_threshold=0.85,
        confirmation_time_sec=0.8,
        torso_angle_threshold=40.0,
        aspect_ratio_threshold=0.85
    )

    results = []
    start_total_time = time.perf_counter()

    for idx, row in df_fps.reset_index(drop=True).iterrows():
        video_name = str(row[col_video]).strip()
        video_path = nofall_dir / video_name

        if not video_path.exists():
            # Probar sin o con _resized si no coincide exacto
            alt_name = video_name.replace("_resized", "") if "_resized" in video_name else f"{Path(video_name).stem}_resized.mp4"
            if (nofall_dir / alt_name).exists():
                video_path = nofall_dir / alt_name
            else:
                continue

        cap = cv2.VideoCapture(str(video_path))
        fps_nominal = cap.get(cv2.CAP_PROP_FPS)
        if fps_nominal <= 0 or np.isnan(fps_nominal) or fps_nominal > 120:
            fps_nominal = 30.0
        dt = 1.0 / fps_nominal

        simulated_ts = 0.0
        fall_confirmed = False
        max_tflite = 0.0
        min_torso = 90.0
        max_ratio = 0.0

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
            if res.aspect_ratio > max_ratio:
                max_ratio = res.aspect_ratio

            if res.state == FallState.CAIDA_CONFIRMADA:
                fall_confirmed = True

        cap.release()
        detector.reset()

        # Extraer prefijo de actividad (C=Chair, W=Walking, B=Bed, R=Recovery, S=Standing)
        activity_prefix = video_name.split("_")[0] if "_" in video_name else video_name[:2]

        # Resultado de la auditoría:
        # Si fall_confirmed es False -> Éxito: el falso positivo fue ELIMINADO
        # Si fall_confirmed es True  -> Persiste el falso positivo
        status = "❌ PERSISTE FP" if fall_confirmed else "✅ FP ELIMINADO"

        results.append({
            "video": video_name,
            "activity": activity_prefix,
            "max_tflite": max_tflite,
            "min_torso": min_torso,
            "max_aspect": max_ratio,
            "confirmed_fall": fall_confirmed,
            "audit_status": status
        })

        print(f"[{idx+1:3d}/{total_fps_original}] {video_name:<24} | TFLite: {max_tflite*100:4.1f}% | Torso: {min_torso:4.1f}° | {status}")

    elapsed = time.perf_counter() - start_total_time
    df_out = pd.DataFrame(results)

    # 4. Métricas Finales
    total_audited = len(df_out)
    fp_eliminated = len(df_out[~df_out["confirmed_fall"]])
    fp_persistent = len(df_out[df_out["confirmed_fall"]])
    reduction_pct = (fp_eliminated / max(total_audited, 1)) * 100.0

    print("\n" + "=" * 70)
    print("📊 RESULTADOS FINALES DE LA AUDITORÍA")
    print("=" * 70)
    print(f"Total videos evaluados:          {total_audited}")
    print(f"Falsos Positivos ELIMINADOS (TN): {fp_eliminated}  ({reduction_pct:.1f}%)")
    print(f"Falsos Positivos PERSISTENTES:   {fp_persistent}  ({100.0 - reduction_pct:.1f}%)")
    print(f"Tiempo total de ejecución:       {elapsed:.1f} s ({total_audited / max(elapsed, 1e-4):.1f} videos/s)")

    # Desglose por actividad
    print("\n📋 Desglose por tipo de actividad:")
    breakdown = df_out.groupby("activity")["confirmed_fall"].agg(
        Total="count",
        Persisten="sum",
        Eliminados=lambda x: (x == False).sum()
    )
    breakdown["Tasa_Supresion"] = (breakdown["Eliminados"] / breakdown["Total"]) * 100.0
    print(breakdown.to_string())
    print("=" * 70)

    # Guardar reporte
    out_csv = project_root / "tests" / "audit_fp_report.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Reporte detallado guardado en: {out_csv.resolve()}\n")


if __name__ == "__main__":
    run_fp_audit()