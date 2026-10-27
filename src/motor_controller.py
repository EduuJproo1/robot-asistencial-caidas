"""
Control de locomoción del robot asistencial (2WD).

Dos servos de rotación continua.
Simulación por defecto, sin acceso a GPIO.

La desactivación de PWM no garantiza frenado físico.
"""

from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic


@dataclass(frozen=True)
class MotorConfig:
    # Numeración BCM
    pin_izq: int = 12
    pin_der: int = 13

    frecuencia_hz: int = 50

    # Punto de cambio de sentido observado experimentalmente.
    # NO corresponde a un punto de parada.
    cruce_us: int = 1600

    # Valores provisionales, pendientes de validar.
    offset_min_us: int = 90
    offset_max_us: int = 250

    # Orientación de los motores según el montaje de referencia.
    sentido_izq: int = -1
    sentido_der: int = 1

    watchdog_s: float = 0.4


class MotorController:

    def __init__(
        self,
        *,
        simulado: bool = True,
        config: MotorConfig | None = None
    ):
        self.cfg = config or MotorConfig()
        self.simulado = simulado

        self._lock = Lock()
        self._cerrado = False
        self._ultimo_comando = monotonic()
        self._moviendo = False
        self._estado = (0, 0)

        self._finalizar = Event()

        self._izq = None
        self._der = None
        self._factory = None

        if not simulado:
            # Usar únicamente después de verificar la
            # protección y alimentación del hardware.
            from gpiozero import PWMOutputDevice
            from gpiozero.pins.lgpio import LGPIOFactory

            self._factory = LGPIOFactory()

            try:
                self._izq = PWMOutputDevice(
                    self.cfg.pin_izq,
                    frequency=self.cfg.frecuencia_hz,
                    initial_value=0.0,
                    pin_factory=self._factory
                )

                self._der = PWMOutputDevice(
                    self.cfg.pin_der,
                    frequency=self.cfg.frecuencia_hz,
                    initial_value=0.0,
                    pin_factory=self._factory
                )

            except Exception:
                if self._izq is not None:
                    self._izq.close()

                self._factory.close()
                raise

        self._vigilante = Thread(
            target=self._vigilar,
            daemon=True
        )
        self._vigilante.start()

    def _duty(self, velocidad: int, sentido: int) -> float:
        if velocidad == 0:
            # Sin pulsos PWM.
            return 0.0

        amplitud = self.cfg.offset_min_us + (
            self.cfg.offset_max_us - self.cfg.offset_min_us
        ) * abs(velocidad) / 100.0

        signo = 1 if velocidad * sentido > 0 else -1

        pulso_us = self.cfg.cruce_us + signo * amplitud

        periodo_us = 1_000_000 / self.cfg.frecuencia_hz

        return pulso_us / periodo_us

    def _aplicar(self, izquierda: int, derecha: int) -> None:
        if (izquierda, derecha) == self._estado:
            return

        self._estado = (izquierda, derecha)

        if self.simulado:
            print(
                f"[SIMULACION] "
                f"Izq={izquierda:+d}% "
                f"Der={derecha:+d}%"
            )
        else:
            self._izq.value = self._duty(
                izquierda, self.cfg.sentido_izq
            )
            self._der.value = self._duty(
                derecha, self.cfg.sentido_der
            )

    def ruedas(self, izquierda: int, derecha: int) -> None:
        izquierda = max(-100, min(100, int(izquierda)))
        derecha = max(-100, min(100, int(derecha)))

        with self._lock:
            if self._cerrado:
                raise RuntimeError("El controlador está cerrado")

            self._ultimo_comando = monotonic()
            self._moviendo = izquierda != 0 or derecha != 0

            self._aplicar(izquierda, derecha)

    def adelante(self, velocidad: int = 30) -> None:
        self.ruedas(velocidad, velocidad)

    def atras(self, velocidad: int = 30) -> None:
        self.ruedas(-velocidad, -velocidad)

    def izquierda(self, velocidad: int = 30) -> None:
        self.ruedas(0, velocidad)

    def derecha(self, velocidad: int = 30) -> None:
        self.ruedas(velocidad, 0)

    def detener(self) -> None:
        self.ruedas(0, 0)

    def _vigilar(self) -> None:
        while not self._finalizar.wait(0.05):

            with self._lock:
                tiempo_sin_orden = (
                    monotonic() - self._ultimo_comando
                )

                if (
                    self._moviendo
                    and tiempo_sin_orden > self.cfg.watchdog_s
                ):
                    self._moviendo = False
                    self._aplicar(0, 0)

                    print(
                        "[WATCHDOG] PWM desactivado "
                        "por falta de órdenes"
                    )

    def cerrar(self) -> None:
        self._finalizar.set()
        self._vigilante.join(timeout=1.0)

        with self._lock:
            if self._cerrado:
                return

            self._moviendo = False
            self._aplicar(0, 0)

            if self._izq is not None:
                self._izq.value = 0.0
                self._izq.close()

            if self._der is not None:
                self._der.value = 0.0
                self._der.close()

            if self._factory is not None:
                self._factory.close()

            self._cerrado = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cerrar()