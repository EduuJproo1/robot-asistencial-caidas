"""
Lógica de navegación del robot asistencial.

Recibe información visual y de proximidad.
No accede a GPIO ni mueve motores directamente.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NavigationCommand:
    accion: str
    velocidad: int = 0
    motivo: str = ""


class NavigationController:

    def __init__(
        self,
        angulo_centrado_deg: float = 10.0,
        distancia_minima_cm: float = 45.0,
        velocidad_avance: int = 20,
        velocidad_giro: int = 20
    ):
        # Valores iniciales para simulación.
        self.angulo_centrado_deg = angulo_centrado_deg
        self.distancia_minima_cm = distancia_minima_cm
        self.velocidad_avance = velocidad_avance
        self.velocidad_giro = velocidad_giro

    def decidir(
        self,
        *,
        objetivo_valido: bool,
        angulo_deg: float | None,
        distancia_obstaculo_cm: float | None = None,
        emergencia_activa: bool = False
    ) -> NavigationCommand:

        # Prioridad 1: detenerse durante una emergencia.
        if emergencia_activa:
            return NavigationCommand(
                "DETENER", motivo="Emergencia activa"
            )

        # Prioridad 2: pérdida de la persona.
        if not objetivo_valido:
            return NavigationCommand(
                "DETENER", motivo="Objetivo no detectado"
            )

        # Prioridad 3: ángulo inválido.
        if angulo_deg is None or not math.isfinite(angulo_deg):
            return NavigationCommand(
                "DETENER", motivo="Orientación no válida"
            )

        # Prioridad 4: sensor no disponible.
        if (
            distancia_obstaculo_cm is None
            or not math.isfinite(distancia_obstaculo_cm)
            or distancia_obstaculo_cm <= 0
        ):
            return NavigationCommand(
                "DETENER", motivo="Sin lectura válida del sensor"
            )

        # Prioridad 5: objeto demasiado próximo.
        if distancia_obstaculo_cm <= self.distancia_minima_cm:
            return NavigationCommand(
                "DETENER", motivo="Obstáculo próximo"
            )

        # Objetivo a la izquierda de la cámara.
        if angulo_deg < -self.angulo_centrado_deg:
            return NavigationCommand(
                "IZQUIERDA", self.velocidad_giro,
                "Objetivo hacia la izquierda"
            )

        # Objetivo a la derecha de la cámara.
        if angulo_deg > self.angulo_centrado_deg:
            return NavigationCommand(
                "DERECHA", self.velocidad_giro,
                "Objetivo hacia la derecha"
            )

        # Objetivo aproximadamente centrado.
        return NavigationCommand(
            "ADELANTE", self.velocidad_avance,
            "Objetivo centrado y frente despejado"
        )