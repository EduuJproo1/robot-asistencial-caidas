"""Benchmark controlado de Vosk para el proyecto robot-asistencial-caidas.

Ubicacion prevista: tests/benchmark_vosk.py (ejecutar desde la raiz del repositorio).
No envia alertas. Reutiliza las listas de palabras clave de VoiceManager.

Ejemplo:
  python tests/benchmark_vosk.py --condition normal
  python tests/benchmark_vosk.py --list-devices
  python tests/benchmark_vosk.py --device-index 1 --condition normal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000
FRAMES_PER_BUFFER = 4000  # Igual que VoiceManager
DISTANCES = ("0.5", "1", "3", "5")
COMMANDS = (
    ("ayuda", "EMERGENCIA"),
    ("emergencia", "EMERGENCIA"),
    ("socorro", "EMERGENCIA"),
    ("llamar", "EMERGENCIA"),
    ("auxilio", "EMERGENCIA"),
    ("estoy bien", "CANCELAR"),
    ("falsa alarma", "CANCELAR"),
    ("cancelar", "CANCELAR"),
    ("detente", "CANCELAR"),
    ("me equivoqué", "CANCELAR"),
)
FIELDS = (
    "condicion", "distancia_m", "numero", "frase_esperada", "accion_esperada",
    "transcripcion_en_vivo", "transcripcion_solo_al_cerrar", "accion_detectada",
    "frase_exacta_correcta", "accion_correcta", "primer_resultado_desde_inicio_s",
    "tiempo_procesamiento_vosk_s", "duracion_grabacion_s", "archivo_audio",
)


def normalize_phrase(value: str) -> str:
    """Normaliza solamente para comparar la frase (no altera la logica de VoiceManager)."""
    no_accents = "".join(
        c for c in unicodedata.normalize("NFD", value.casefold())
        if unicodedata.category(c) != "Mn"
    )
    return " ".join(no_accents.split())


def classify_like_voice_manager(text: str, voice_manager_class: type) -> str:
    normalized = normalize_phrase(text)
    if any(
        normalize_phrase(kw) in normalized
        for kw in voice_manager_class.KEYWORDS_EMERGENCIA
    ):
        return "EMERGENCIA"
    if any(
        normalize_phrase(kw) in normalized
        for kw in voice_manager_class.KEYWORDS_CANCELACION
    ):
        return "CANCELAR"
    return "DESCONOCIDO"


def list_input_devices(pyaudio_module: Any) -> None:
    audio = pyaudio_module.PyAudio()
    try:
        print("\nDispositivos de entrada disponibles:")
        for i in range(audio.get_device_count()):
            device = audio.get_device_info_by_index(i)
            if int(device.get("maxInputChannels", 0)) > 0:
                print(f"  [{i}] {device['name']} | canales entrada: {device['maxInputChannels']}")
    finally:
        audio.terminate()


def record_live(stream: Any, duration_sec: float, recognizer: Any) -> tuple[bytes, str, str, float | None, float, float]:
    """Graba y decodifica simultaneamente, como el flujo de VoiceManager.

    El resultado de FinalResult() se almacena SOLO como diagnostico, porque
    VoiceManager procesa los resultados de AcceptWaveform() (Result()).
    """
    pieces: list[bytes] = []
    live_segments: list[str] = []
    first_result_sec: float | None = None
    decode_time_sec = 0.0
    expected_chunks = round(duration_sec * SAMPLE_RATE / FRAMES_PER_BUFFER)

    stream.start_stream()
    start = time.perf_counter()
    try:
        for _ in range(expected_chunks):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            pieces.append(data)
            tick = time.perf_counter()
            finished_utterance = recognizer.AcceptWaveform(data)
            if finished_utterance:
                result = json.loads(recognizer.Result())
                phrase = result.get("text", "").strip().lower()
                if phrase:
                    live_segments.append(phrase)
                    if first_result_sec is None:
                        first_result_sec = time.perf_counter() - start
            decode_time_sec += time.perf_counter() - tick
    finally:
        stream.stop_stream()
    real_duration = time.perf_counter() - start

    tick = time.perf_counter()
    final_result = json.loads(recognizer.FinalResult())
    decode_time_sec += time.perf_counter() - tick
    final_tail = final_result.get("text", "").strip().lower()

    return (b"".join(pieces), " ".join(live_segments), final_tail,
            first_result_sec, decode_time_sec, real_duration)


def save_wav(path: Path, raw_audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(raw_audio)


def save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def save_summary(path: Path, records: list[dict[str, Any]]) -> None:
    groups = [(distance, [r for r in records if r["distancia_m"] == distance]) for distance in DISTANCES]
    groups.append(("TOTAL", records))
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.writer(out)
        writer.writerow(("distancia_m", "n", "frases_exactas", "frases_exactas_pct",
                         "acciones_correctas", "acciones_correctas_pct"))
        for name, group in groups:
            n = len(group)
            exact = sum(r["frase_exacta_correcta"] for r in group)
            actions = sum(r["accion_correcta"] for r in group)
            writer.writerow((name, n, exact, f"{100*exact/n:.2f}" if n else "",
                             actions, f"{100*actions/n:.2f}" if n else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba Vosk en 0,5 / 1 / 3 / 5 metros; 10 expresiones cada una")
    parser.add_argument("--condition", choices=("normal",), default="normal",
                        help="Condicion ambiental de la sesion")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models" / "vosk-model-es")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evidences" / "vosk_benchmark")
    parser.add_argument("--device-index", type=int, default=None,
                        help="Indice de microfono. Si no se indica, usa el predeterminado del sistema")
    parser.add_argument("--list-devices", action="store_true", help="Lista microfonos y termina")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="Segundos de captura por ensayo (6 por defecto, incluye silencios)")
    parser.add_argument("--notes", default="", help="Observaciones ambientales")
    args = parser.parse_args()

    try:
        import pyaudio
        from vosk import Model, KaldiRecognizer, SetLogLevel
        from src.voice_manager import VoiceManager
    except ImportError as exc:
        print(f"Falta una dependencia del repositorio: {exc}")
        print("Ejecuta el script desde el mismo entorno Python donde funciona test_vosk_mic.py")
        return 1

    SetLogLevel(-1)
    if args.list_devices:
        list_input_devices(pyaudio)
        return 0
    if args.duration <= 0:
        parser.error("--duration debe ser positiva")
    if not args.model_path.exists():
        print(f"No se encuentra el modelo: {args.model_path}")
        return 1

    session_dir = args.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.condition)
    session_dir.mkdir(parents=True, exist_ok=False)
    model = Model(str(args.model_path))
    audio = pyaudio.PyAudio()
    stream = None
    records: list[dict[str, Any]] = []
    try:
        device_info = (audio.get_device_info_by_index(args.device_index)
                       if args.device_index is not None
                       else audio.get_default_input_device_info())
        print(f"\nMicrófono seleccionado: {device_info['name']}")
        stream = audio.open(
            format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
            input=True, input_device_index=args.device_index,
            frames_per_buffer=FRAMES_PER_BUFFER, start=False,
        )
        metadata = {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "condicion": args.condition, "distancias_m": DISTANCES,
            "mic_name": device_info["name"], "mic_index": device_info["index"],
            "model_path": str(args.model_path.resolve()), "sample_rate_hz": SAMPLE_RATE,
            "duracion_configurada_s": args.duration, "observaciones": args.notes,
            "criterio": "Frase exacta normalizada y accion detectada por VoiceManager; "
                        "solo Result() de AcceptWaveform() cuenta para el exito en vivo.",
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nCONDICION:", args.condition.upper())
        print("Por ensayo: 6 s de grabacion por defecto, palabra al comienzo y pausa de silencio al final.")
        print("No se enviaran alertas de emergencia. Resultados guardados tras cada ensayo.")

        for distance in DISTANCES:
            print(f"\n{'='*62}\nDISTANCIA: {distance} METROS\n{'='*62}")
            input(f"Ponte en la marca de {distance} m mirando hacia el micrófono. Pulsa Enter al estar listo...")
            for i, (expected, action_expected) in enumerate(COMMANDS, start=1):
                while True:
                    prompt = input(f"\n[{distance} m, {i}/10] Di: '{expected}'. Enter=grabar / q=salir: ").strip().lower()
                    if prompt == "q":
                        raise KeyboardInterrupt
                    for n in (3, 2, 1):
                        print(n, end="... ", flush=True)
                        time.sleep(0.75)
                    print("¡HABLA AHORA!", flush=True)

                    recognizer = KaldiRecognizer(model, SAMPLE_RATE)  # Estado limpio por ensayo
                    raw, live_text, closing_text, first_s, cpu_s, capture_s = record_live(
                        stream, args.duration, recognizer
                    )
                    action_detected = classify_like_voice_manager(live_text, VoiceManager)
                    exact_match = int(normalize_phrase(live_text) == normalize_phrase(expected))
                    action_match = int(action_detected == action_expected)
                    print(f"  Vosk (en vivo): {live_text!r}")
                    if closing_text:
                        print(f"  Solo al cerrar ensayo (NO puntua): {closing_text!r}")
                    print(f"  Frase exacta: {'SI' if exact_match else 'NO'} | Accion: {action_detected} "
                          f"({'CORRECTA' if action_match else 'INCORRECTA'})")
                    choice = input("Enter=guardar ensayo / r=repetir SOLO por problema de grabación / q=salir: ").strip().lower()
                    if choice == "q":
                        raise KeyboardInterrupt
                    if choice == "r":
                        print("Repitiendo por fallo procedimental. No repetir para mejorar el resultado de Vosk.")
                        continue

                    audio_file = session_dir / "audios" / f"{distance.replace('.', '_')}m" / f"{i:02d}_{expected.replace(' ', '_')}.wav"
                    save_wav(audio_file, raw)
                    records.append({
                        "condicion": args.condition, "distancia_m": distance, "numero": i,
                        "frase_esperada": expected, "accion_esperada": action_expected,
                        "transcripcion_en_vivo": live_text,
                        "transcripcion_solo_al_cerrar": closing_text,
                        "accion_detectada": action_detected,
                        "frase_exacta_correcta": exact_match, "accion_correcta": action_match,
                        "primer_resultado_desde_inicio_s": round(first_s, 3) if first_s is not None else "",
                        "tiempo_procesamiento_vosk_s": round(cpu_s, 4),
                        "duracion_grabacion_s": round(capture_s, 3),
                        "archivo_audio": str(audio_file.relative_to(session_dir)),
                    })
                    save_csv(session_dir / "resultados_vosk.csv", records)
                    save_summary(session_dir / "resumen_vosk.csv", records)
                    break
    except KeyboardInterrupt:
        print("\nPrueba interrumpida. Los ensayos previamente guardados permanecen disponibles.")
    except (OSError, ValueError) as exc:
        print(f"\nProblema de audio/dispositivo: {exc}")
        print("Comprueba que el micrófono soporte mono PCM 16-bit a 16.000 Hz, como VoiceManager.")
        return 1
    finally:
        if stream is not None:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        audio.terminate()

    if records:
        total = len(records)
        exact = sum(r["frase_exacta_correcta"] for r in records)
        actions = sum(r["accion_correcta"] for r in records)
        print(f"\n{total} pruebas registradas. Frases exactas: {exact}/{total} ({100*exact/total:.1f} %)")
        print(f"Acciones correctas: {actions}/{total} ({100*actions/total:.1f} %)")
        print("NOTA: El tiempo desde inicio de audio NO es latencia desde el fin del habla.")
        print(f"Evidencias: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
