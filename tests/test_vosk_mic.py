"""
Prueba de Reconocimiento de Voz Offline con Vosk.
Verifica el funcionamiento del micrófono y la detección de palabras clave en español.
"""

import json
import queue
import sys
from pathlib import Path

import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

# Suprimir logs de C++ de Vosk para mantener la consola limpia
SetLogLevel(-1)

# Diccionario de comandos esperados
COMANDOS_EMERGENCIA = ["ayuda", "emergencia", "socorro", "llamar"]
COMANDOS_CANCELACION = ["estoy bien", "falsa alarma", "cancelar", "detente"]

def test_microphone():
    model_path = Path("models/vosk-model-es")
    if not model_path.exists():
        print(f"❌ Error: No se encontró el modelo de voz en {model_path.resolve()}")
        print("Por favor, descarga vosk-model-small-es y extráelo en esa ruta.")
        sys.exit(1)

    print("Cargando modelo acústico en español (offline)...")
    model = Model(str(model_path))
    
    # Tasa de muestreo estándar de 16kHz requerida por Vosk
    samplerate = 16000
    recognizer = KaldiRecognizer(model, samplerate)

    audio_queue = queue.Queue()

    def callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    
    print("\nDetectando micrófono predeterminado...")
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=samplerate,
            input=True,
            frames_per_buffer=4000,
            stream_callback=callback
        )
    except Exception as e:
        print(f"❌ Error al abrir el micrófono: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("🎙️ SISTEMA DE VOZ ACTIVO (Prueba)")
    print("="*50)
    print("Di algo en español... (Presiona Ctrl+C para salir)")
    print("Palabras clave a probar:")
    print(f" - Emergencia: {COMANDOS_EMERGENCIA}")
    print(f" - Cancelación: {COMANDOS_CANCELACION}\n")

    stream.start_stream()

    try:
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                texto = result.get("text", "").strip()
                
                if texto:
                    print(f"🗣️ Escuchado: '{texto}'")
                    
                    # Evaluar si el texto contiene palabras clave
                    if any(cmd in texto for cmd in COMANDOS_EMERGENCIA):
                        print("   🚨 -> COMANDO DE EMERGENCIA DETECTADO")
                    elif any(cmd in texto for cmd in COMANDOS_CANCELACION):
                        print("   ✅ -> COMANDO DE CANCELACIÓN DETECTADO")
                        
    except KeyboardInterrupt:
        print("\nDeteniendo prueba de micrófono...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    test_microphone()