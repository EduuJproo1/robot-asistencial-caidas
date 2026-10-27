
"""Benchmark de 10 alertas Telegram mediante AlertManager."""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue
from statistics import mean, median
from time import monotonic, perf_counter, sleep
from unittest.mock import patch

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alert_manager import AlertManager


NUM_PRUEBAS = 10
COOLDOWN_S = 20.0
CARPETA = ROOT / "evidences" / "benchmark_telegram_rpi"


class AlertManagerMedido(AlertManager):
    def __init__(self, **kwargs):
        super().__init__(max_dispatch_workers=1, **kwargs)
        self.pendientes = Queue()
        self.finalizados = Queue()

    def _send_telegram_worker(self, message, photo_path):
        numero, instante_evento = self.pendientes.get()

        registro = {
            "ensayo": numero,
            "api_aceptada": False,
            "http_final": "",
            "intentos_http": 0,
            "latencia_api_s": "",
            "error": "",
        }

        post_original = requests.post

        def post_medido(*args, **kwargs):
            registro["intentos_http"] += 1

            try:
                respuesta = post_original(*args, **kwargs)
            except requests.RequestException as exc:
                registro["error"] = type(exc).__name__
                raise

            registro["http_final"] = respuesta.status_code

            try:
                aceptada = bool(
                    respuesta.ok and respuesta.json().get("ok")
                )
            except (ValueError, AttributeError):
                aceptada = False

            if aceptada:
                registro["api_aceptada"] = True
                registro["latencia_api_s"] = round(
                    perf_counter() - instante_evento, 3
                )
                registro["error"] = ""

            return respuesta

        aviso = (
            f"🧪 *PRUEBA TÉCNICA {numero:02d}/{NUM_PRUEBAS}*\n"
            "*NO ES UNA EMERGENCIA REAL*\n\n"
        )

        try:
            # Se conserva el worker original de AlertManager.
            # Solo se instrumenta temporalmente requests.post.
            with patch.object(
                requests, "post", side_effect=post_medido
            ):
                super()._send_telegram_worker(
                    aviso + message, photo_path
                )

        except Exception as exc:
            registro["error"] = repr(exc)

        finally:
            self.finalizados.put(registro)


def main():
    load_dotenv(ROOT / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise SystemExit(
            "Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID"
        )

    CARPETA.mkdir(parents=True, exist_ok=True)

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = CARPETA / f"telegram_{marca}.csv"

    # Imagen sintética: no requiere encender la webcam.
    imagen = np.full((480, 800, 3), 245, dtype=np.uint8)

    cv2.putText(
        imagen, "PRUEBA TECNICA", (95, 195),
        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (25, 25, 25), 3
    )
    cv2.putText(
        imagen, "NO ES UNA EMERGENCIA REAL", (80, 260),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 180), 2
    )

    filas = []
    ultimo_evento = None

    gestor = AlertManagerMedido(
        bot_token=token,
        chat_id=chat_id,
        cooldown_sec=COOLDOWN_S,
        evidence_dir=str(CARPETA / "fotos"),
    )

    print("Se enviarán 10 fotografías de prueba.")
    print("Observa el chat de Telegram.")
    print("Pulsa ENTER inmediatamente al ver cada mensaje.")
    print("Si no llega, escribe 'n' tras esperar al menos 30 s.")

    input("\nPulsa ENTER para comenzar...")

    try:
        for numero in range(1, NUM_PRUEBAS + 1):

            # Respetar el cooldown de 20 s.
            if ultimo_evento is not None:
                espera = (
                    COOLDOWN_S + 0.3
                    - (monotonic() - ultimo_evento)
                )

                if espera > 0:
                    print(
                        f"Esperando {espera:.1f} s "
                        "por cooldown..."
                    )
                    sleep(espera)

            fecha = datetime.now().isoformat(
                timespec="seconds"
            )

            # Inicio de la medición del evento simulado.
            t_evento = perf_counter()
            ultimo_evento = monotonic()

            gestor.pendientes.put((numero, t_evento))

            encolada = gestor.trigger_fall_alert(
                frame_bgr=imagen,
                fall_confidence=0.0,
                torso_angle=0.0,
                time_in_fall=0.0,
                emergency_by_voice=True
            )

            if not encolada:
                raise RuntimeError(
                    "Alerta bloqueada por cooldown."
                )

            print(
                f"\nEnsayo {numero}/{NUM_PRUEBAS}: "
                "alerta encolada."
            )

            respuesta = input(
                "ENTER al verla en Telegram; "
                "'n' si no llegó: "
            ).strip().lower()

            t_observacion = perf_counter()

            recibida = respuesta in (
                "", "s", "si", "sí"
            )

            estado = gestor.finalizados.get(timeout=45)

            estado.update({
                "fecha": fecha,
                "encolada": encolada,
                "recibida_observada": recibida,
                "latencia_observada_s": (
                    round(t_observacion - t_evento, 3)
                    if recibida else ""
                ),
            })

            filas.append(estado)

            print(
                f"API aceptada: {estado['api_aceptada']} | "
                f"API: {estado['latencia_api_s']} s | "
                f"Observada: "
                f"{estado['latencia_observada_s']} s"
            )

    finally:
        gestor.shutdown(wait=True)

        columnas = [
            "ensayo",
            "fecha",
            "encolada",
            "api_aceptada",
            "recibida_observada",
            "latencia_api_s",
            "latencia_observada_s",
            "intentos_http",
            "http_final",
            "error"
        ]

        with destino.open(
            "w", newline="", encoding="utf-8"
        ) as archivo:
            writer = csv.DictWriter(
                archivo, fieldnames=columnas
            )
            writer.writeheader()
            writer.writerows(filas)

        api = [
            r["latencia_api_s"]
            for r in filas
            if r["api_aceptada"]
        ]

        observadas = [
            r["latencia_observada_s"]
            for r in filas
            if r["recibida_observada"]
        ]

        print("\n===== RESULTADOS TELEGRAM =====")
        print(
            f"API aceptadas: {len(api)}/{len(filas)}"
        )
        print(
            f"Recepciones observadas: "
            f"{len(observadas)}/{len(filas)}"
        )

        if api:
            print(
                f"Latencia API: "
                f"media {mean(api):.3f} s | "
                f"mediana {median(api):.3f} s"
            )

        if observadas:
            print(
                f"Latencia observada: "
                f"media {mean(observadas):.3f} s | "
                f"mediana {median(observadas):.3f} s"
            )

        print(f"CSV guardado en: {destino}")


if __name__ == "__main__":
    main()
