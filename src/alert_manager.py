"""
Módulo de Gestión de Alertas y Despacho de Eventos Críticos.
Gestiona el almacenamiento de evidencias locales, el control de spam (cooldown)
y el despacho asíncrono y acotado de notificaciones hacia Telegram.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Optional

import cv2
import requests


class AlertManager:
    """
    Concurrencia: el cooldown se protege con un lock explícito para evitar una
    condición de carrera check-then-act sobre `_last_alert_time` si en el
    futuro `trigger_fall_alert` se invoca desde más de un hilo (p. ej. soporte
    multi-cámara). El despacho de red corre en un ThreadPoolExecutor acotado
    (en vez de un `Thread` nuevo por alerta) para permitir un apagado
    ordenado del proceso y evitar que hilos colgados por timeout de red se
    acumulen sin control.
    """

    MAX_SEND_RETRIES: int = 2
    RETRY_BACKOFF_SEC: float = 1.5
    REQUEST_TIMEOUT_SEC: float = 8.0

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        cooldown_sec: float = 30.0,
        evidence_dir: str = "evidences",
        max_dispatch_workers: int = 2,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_sec = cooldown_sec

        self._last_alert_time: float = 0.0
        self._cooldown_lock = threading.Lock()

        self.evidence_path = Path(evidence_dir).resolve()
        self.evidence_path.mkdir(parents=True, exist_ok=True)

        self._executor = ThreadPoolExecutor(
            max_workers=max_dispatch_workers, thread_name_prefix="alert-dispatch"
        )

    def _send_telegram_worker(self, message: str, photo_path: Optional[Path]) -> None:
        """Ejecuta en el pool de despacho para no bloquear el procesamiento de video."""
        if not self.bot_token or not self.chat_id:
            return

        has_photo = bool(photo_path and photo_path.exists())
        url = (
            f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            if has_photo
            else f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        )

        total_attempts = self.MAX_SEND_RETRIES + 1
        for attempt in range(1, total_attempts + 1):
            try:
                if has_photo:
                    with open(photo_path, "rb") as photo_file:
                        response = requests.post(
                            url,
                            data={"chat_id": self.chat_id, "caption": message, "parse_mode": "Markdown"},
                            files={"photo": photo_file},
                            timeout=self.REQUEST_TIMEOUT_SEC,
                        )
                else:
                    response = requests.post(
                        url,
                        data={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"},
                        timeout=self.REQUEST_TIMEOUT_SEC,
                    )
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                is_last_attempt = attempt == total_attempts
                suffix = "se agotaron los reintentos." if is_last_attempt else "reintentando..."
                print(f"⚠️ [AlertManager] Intento {attempt}/{total_attempts} de despacho falló: {exc} — {suffix}")
                if not is_last_attempt:
                    time.sleep(self.RETRY_BACKOFF_SEC * attempt)

    def trigger_fall_alert(
        self,
        frame_bgr,
        fall_confidence: float,
        torso_angle: float,
        time_in_fall: float,
    ) -> bool:
        """
        Dispara una alerta de emergencia si transcurrió el tiempo de enfriamiento.
        Retorna True si la alerta fue encolada para despacho, False si fue
        bloqueada por cooldown.
        """
        now = time.monotonic()
        with self._cooldown_lock:
            if (now - self._last_alert_time) < self.cooldown_sec:
                return False
            self._last_alert_time = now

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Guardar evidencia en disco
        image_filename = self.evidence_path / f"fall_{timestamp_str}.jpg"
        cv2.imwrite(str(image_filename), frame_bgr)

        # 2. Estructurar mensaje
        message = (
            "🚨 *ALERTA CRÍTICA: CAÍDA CONFIRMADA*\n\n"
            f"📅 *Hora:* `{datetime.now().strftime('%H:%M:%S')}`\n"
            f"🎯 *Certeza del Impacto:* `{fall_confidence * 100:.1f}%`\n"
            f"📐 *Inclinación del Torso:* `{torso_angle:.1f}°`\n"
            f"⏱️ *Tiempo en el Suelo:* `{time_in_fall:.1f} s`\n\n"
            "⚠️ _El robot asistencial solicita verificación inmediata._"
        )

        print(f"\n📢 [AlertManager] ¡EMERGENCIA EMITIDA! Evidencia guardada en: {image_filename.name}")

        # 3. Despachar notificación de forma asíncrona y acotada
        self._executor.submit(self._send_telegram_worker, message, image_filename)

        return True

    def shutdown(self, wait: bool = True) -> None:
        """
        Cierra el pool de despacho de forma ordenada. Debe llamarse al
        finalizar el pipeline (idealmente en un bloque `finally`) para no
        dejar hilos de red colgados al cerrar el proceso.
        """
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
