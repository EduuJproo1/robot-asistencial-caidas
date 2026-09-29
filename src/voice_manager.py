"""
Módulo de Interacción por Voz (STT Offline).
Ejecuta el reconocimiento acústico de Vosk en un hilo secundario y
reporta palabras clave detectadas mediante una cola thread-safe.
"""

import json
import queue
import threading
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

# Ocultar logs de C++ de Kaldi/Vosk para mantener limpia la consola
SetLogLevel(-1)

class VoiceCommand(Enum):
    EMERGENCIA = auto()
    CANCELAR = auto()
    DESCONOCIDO = auto()


class VoiceManager:
    # Diccionarios de palabras clave (Keywords)
    KEYWORDS_EMERGENCIA = {"ayuda", "emergencia", "socorro", "llamar", "auxilio"}
    KEYWORDS_CANCELACION = {"estoy bien", "falsa alarma", "cancelar", "detente", "me equivoque"}

    def __init__(self, model_path: str = "models/vosk-model-es"):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo de voz no encontrado en: {self.model_path}")

        self.command_queue: queue.Queue[VoiceCommand] = queue.Queue()
        self._is_running = False
        self._audio_thread: Optional[threading.Thread] = None

        # Inicialización diferida (se cargan al arrancar el hilo)
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None

    def start(self) -> None:
        """Inicia el reconocimiento de voz en un hilo secundario (Daemon)."""
        if self._is_running:
            return

        self._is_running = True
        self._audio_thread = threading.Thread(
            target=self._audio_processing_loop,
            name="VoiceRecognitionThread",
            daemon=True  # Se cierra automáticamente si el hilo principal muere
        )
        self._audio_thread.start()

    def stop(self) -> None:
        """Detiene la captura y libera los recursos de audio."""
        self._is_running = False
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)

        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()

    def get_latest_command(self) -> Optional[VoiceCommand]:
        """
        Extrae el último comando de la cola de forma no bloqueante.
        Retorna None si no hay comandos pendientes.
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def _audio_processing_loop(self) -> None:
        """Bucle interno del hilo de reconocimiento acústico."""
        print(f"🎙️ [VoiceManager] Cargando modelo acústico desde {self.model_path.name}...")
        self.model = Model(str(self.model_path))
        samplerate = 16000
        self.recognizer = KaldiRecognizer(self.model, samplerate)

        # Cola interna para pasar audio del callback de PyAudio al reconocedor
        raw_audio_queue = queue.Queue()

        def audio_callback(in_data, frame_count, time_info, status):
            if self._is_running:
                raw_audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        self.audio_interface = pyaudio.PyAudio()
        try:
            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=samplerate,
                input=True,
                frames_per_buffer=4000,
                stream_callback=audio_callback
            )
            self.audio_stream.start_stream()
            print("🎙️ [VoiceManager] Sistema de voz activo y escuchando comandos.")
        except Exception as e:
            print(f"❌ [VoiceManager] Error al abrir el micrófono: {e}")
            self._is_running = False
            return

        while self._is_running:
            try:
                # Timeout corto para permitir que el bucle revise _is_running
                data = raw_audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()

                if text:
                    self._process_text(text)

    def _process_text(self, text: str) -> None:
        """Clasifica el texto reconocido y encola el comando correspondiente."""
        # Se busca si alguna palabra clave está contenida en la frase escuchada
        if any(kw in text for kw in self.KEYWORDS_EMERGENCIA):
            print(f"\n🗣️ [VoiceManager] Frase: '{text}' -> DISPARA EMERGENCIA")
            self.command_queue.put(VoiceCommand.EMERGENCIA)
        
        elif any(kw in text for kw in self.KEYWORDS_CANCELACION):
            print(f"\n🗣️ [VoiceManager] Frase: '{text}' -> DISPARA CANCELACIÓN")
            self.command_queue.put(VoiceCommand.CANCELAR)