"""
Define los parámetros por defecto del subsistema de detección.

Los valores pueden ser sobreescritos explícitamente al construir
DetectorConfig para una ejecución o experimento determinado.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuración centralizada del detector de caídas.

    Los valores definidos aquí controlan los umbrales geométricos,
    temporización de la FSM, suavizado y parámetros de cámara.
    """
    # 1. Parámetros Cinemáticos (Transformer TFLite)
    tflite_conf_threshold: float = 0.85      # Sensibilidad para capturar caídas amortiguadas
    feature_window_size: int = 30            # Fotogramas de entrada (1.0 s a 30 FPS)

    # 2. Umbrales geométricos obtenidos mediante calibración preliminar
    torso_angle_threshold: float = 40.0      # Ángulo torso-suelo (°). <= => torso horizontal (suelo)
    aspect_ratio_threshold: float = 0.85     # Bounding box ancho/alto. >= => silueta horizontal
    min_keypoint_confidence: float = 0.25    # Filtro de visibilidad MediaPipe

    # 3. Restricciones Biomecánicas (Antropometría y Posturas Irregulares)
    height_collapse_ratio: float = 0.45      # Altura actual <= 45% de altura bípeda -> colapso (suelo)
    seated_height_ratio_max: float = 0.75    # Banda (height_collapse_ratio, este valor] -> candidato sentado
    upright_torso_angle: float = 55.0        # Ángulo mínimo de torso para considerar bipedestación
    shoulder_high_position_ratio: float = 0.45
    # Posición Y normalizada de los hombros por debajo de la cual se asume postura erguida
    # aun con ángulo de torso ambiguo (hombros ocupando el tercio superior del cuadro).
    # Se reutiliza tanto con torso completo visible como con caderas fuera de cuadro,
    # evitando dos literales iguales dispersos en el código (0.45 en ambos casos).

    # 4. Suavizado temporal (EMA) — aplica a métricas posturales y de navegación
    smoothing_alpha: float = 0.65            # Ponderación del valor actual vs. histórico

    # 5. Temporización y FSM (Requerimiento RF3)
    confirmation_time_sec: float = 1.2       # Persistencia en el suelo para confirmar caída
    ambiguous_timeout_sec: float = 3.0       # Máx. tiempo en CONFIRMANDO sin evidencia clara -> descarta
    recovery_hold_sec: float = 1.0           # Postura de recuperación sostenida para anular emergencia
    post_impact_grace_sec: float = 0.8       # Tolerancia ante oclusión/ruido de MediaPipe en el suelo

    # 6. Parámetros de Cámara y Guiado del Robot
    camera_hfov_deg: float = 60.0            # Campo de visión horizontal nominal (calibrar por cámara)
    assumed_person_height_m: float = 1.65    # Altura antropométrica promedio, para estimar distancia

    def __post_init__(self) -> None:
        if not (0.0 < self.height_collapse_ratio < self.seated_height_ratio_max < 1.0):
            raise ValueError(
                "Se requiere 0 < height_collapse_ratio < seated_height_ratio_max < 1"
            )
        if not (0.0 < self.camera_hfov_deg < 180.0):
            raise ValueError("camera_hfov_deg debe estar en (0, 180) grados")
        if not (0.0 < self.smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha debe estar en (0, 1]")
        if self.ambiguous_timeout_sec <= 0.0:
            raise ValueError("ambiguous_timeout_sec debe ser positivo")
        if self.assumed_person_height_m <= 0.0:
            raise ValueError("assumed_person_height_m debe ser positivo")
