
"""
Benchmark de FPS del pipeline integrado en Raspberry Pi 5.
Calentamiento: 5 segundos.
Medición: 180 segundos.
Telegram desactivado.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main

CALENTAMIENTO_S = 5.0
TIEMPO_MEDICION_S = 180.0

inicio_total = None
inicio_medicion = None
fin_medicion = None
muestras = []


class BenchmarkTerminado(Exception):
    """Señal para detener el pipeline al terminar la medición."""


dibujar_original = main.draw_hud_and_telemetry


def dibujar_y_medir(frame, result, fps, confirm_sec):
    global inicio_medicion, fin_medicion

    ahora = perf_counter()
    tiempo_total = ahora - inicio_total

    # Fase 1: calentamiento sin registrar muestras.
    if tiempo_total < CALENTAMIENTO_S:
        return dibujar_original(frame, result, fps, confirm_sec)

    # Comienza la medición al terminar el calentamiento.
    if inicio_medicion is None:
        inicio_medicion = ahora
        print("\nCalentamiento completado. Iniciando medición de 180 s...")

    tiempo_medicion = ahora - inicio_medicion

    # Finalizar cuando hayan transcurrido 180 segundos de medición.
    if tiempo_medicion >= TIEMPO_MEDICION_S:
        fin_medicion = ahora
        raise BenchmarkTerminado

    # Registrar exclusivamente las muestras de la medición.
    muestras.append((tiempo_medicion, fps))

    return dibujar_original(frame, result, fps, confirm_sec)


main.draw_hud_and_telemetry = dibujar_y_medir

try:
    print("Benchmark FPS: visión + Vosk")
    print("Calentamiento: 5 segundos")
    print("Medición: 180 segundos")
    print("Telegram desactivado")
    print("Iniciando pipeline...")

    inicio_total = perf_counter()

    main.run_pipeline(
        source=0,
        vision_only=False,
        no_telegram=True,
        width=1280,
        height=720
    )

except BenchmarkTerminado:
    print("\nTiempo de medición alcanzado.")

finally:
    main.draw_hud_and_telemetry = dibujar_original

    if inicio_medicion is not None:
        if fin_medicion is None:
            fin_medicion = perf_counter()

        duracion = fin_medicion - inicio_medicion

        if len(muestras) >= 2:
            cantidad = len(muestras)

            # Media aritmética de los FPS mostrados por el HUD.
            fps_promedio = sum(fps for _, fps in muestras) / cantidad

            # Intervalo entre la primera y la última muestra.
            intervalo_muestras = muestras[-1][0] - muestras[0][0]
            fps_efectivos = (
                (cantidad - 1) / intervalo_muestras
                if intervalo_muestras > 0
                else 0.0
            )

            carpeta = ROOT / "evidences" / "benchmark_fps_rpi"
            carpeta.mkdir(parents=True, exist_ok=True)

            nombre = (
                "fps_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )

            destino = carpeta / nombre

            with destino.open(
                "w", newline="", encoding="utf-8"
            ) as archivo:
                writer = csv.writer(archivo)
                writer.writerow([
                    "tiempo_s",
                    "fps_instantaneo_hud"
                ])
                writer.writerows(muestras)

            print("\n===== RESULTADOS FPS =====")
            print(f"Duración de medición: {duracion:.2f} s")
            print(f"Muestras registradas: {cantidad}")
            print(f"FPS promedio HUD: {fps_promedio:.2f}")
            print(f"FPS efectivos entre muestras: {fps_efectivos:.2f}")
            print(f"CSV guardado en: {destino}")

        else:
            print("No se obtuvieron suficientes muestras.")

    else:
        print("El pipeline terminó antes de completar el calentamiento.")