"""
Módulo de detección de caídas y generación de telemetría visual.

Integra:
1. Inferencia temporal mediante el modelo TFLite seleccionado en la Etapa 1.
2. Análisis geométrico de postura a partir de landmarks de MediaPipe Pose.
3. Suavizado EMA de métricas posturales y de navegación.
4. Máquina de estados para confirmación, descarte y recuperación de caídas.
5. Estimación de orientación y distancia para el futuro control de la base móvil.

La coordenada z de MediaPipe se utiliza únicamente como información de
profundidad relativa para complementar el cálculo geométrico del torso; no
se interpreta como una distancia métrica.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf

from src.detector_config import DetectorConfig

Interpreter = tf.lite.Interpreter

# --------------------------------------------------------------------------- #
# Esquema de keypoints — fijo para MediaPipe Pose (17 puntos). Se calcula una
# única vez a nivel de módulo: no hay razón para reconstruir este mapeo por
# cada instancia de FallDetector.
# --------------------------------------------------------------------------- #
_POSE_LANDMARK = mp.solutions.pose.PoseLandmark

_LANDMARK_ENUM_TO_NAME: Dict[int, str] = {
    _POSE_LANDMARK.NOSE: "Nose",
    _POSE_LANDMARK.LEFT_EYE: "Left Eye",
    _POSE_LANDMARK.RIGHT_EYE: "Right Eye",
    _POSE_LANDMARK.LEFT_EAR: "Left Ear",
    _POSE_LANDMARK.RIGHT_EAR: "Right Ear",
    _POSE_LANDMARK.LEFT_SHOULDER: "Left Shoulder",
    _POSE_LANDMARK.RIGHT_SHOULDER: "Right Shoulder",
    _POSE_LANDMARK.LEFT_ELBOW: "Left Elbow",
    _POSE_LANDMARK.RIGHT_ELBOW: "Right Elbow",
    _POSE_LANDMARK.LEFT_WRIST: "Left Wrist",
    _POSE_LANDMARK.RIGHT_WRIST: "Right Wrist",
    _POSE_LANDMARK.LEFT_HIP: "Left Hip",
    _POSE_LANDMARK.RIGHT_HIP: "Right Hip",
    _POSE_LANDMARK.LEFT_KNEE: "Left Knee",
    _POSE_LANDMARK.RIGHT_KNEE: "Right Knee",
    _POSE_LANDMARK.LEFT_ANKLE: "Left Ankle",
    _POSE_LANDMARK.RIGHT_ANKLE: "Right Ankle",
}
KEYPOINT_NAMES: Tuple[str, ...] = tuple(sorted(_LANDMARK_ENUM_TO_NAME.values()))
KPT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)
FEATURE_DIM: int = NUM_KEYPOINTS * 3  # (x, y, visibilidad) por keypoint -> 51, firma del modelo TFLite


def _robust_midpoint(
    ax: float, ay: float, ac: float,
    bx: float, by: float, bc: float,
    min_confidence: float,
) -> Tuple[float, float, bool]:
    """
    Calcula el punto medio robusto de dos landmarks visibles.
    Si solo uno supera el umbral mínimo de visibilidad, utiliza dicho punto
    como aproximación.
    """
    valid_a = ac > min_confidence
    valid_b = bc > min_confidence
    if valid_a and valid_b:
        return (ax + bx) / 2.0, (ay + by) / 2.0, True
    if valid_a:
        return ax, ay, True
    if valid_b:
        return bx, by, True
    return math.nan, math.nan, False


class ExponentialMovingAverage:
    """Filtro pasa-bajos de primer orden para atenuar ruido de MediaPipe frame a frame."""

    def __init__(self, alpha: float):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha debe estar en (0, 1]")
        self._alpha = alpha
        self._value: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        """Restablece el valor de una instancia de EMA"""
        self._value = None

    def update(self, sample: float) -> float:
        self._value = (
            sample if self._value is None
            else self._alpha * sample + (1.0 - self._alpha) * self._value
        )
        return self._value


class FallState(Enum):
    """Estados posibles de la máquina de confirmación de caídas."""
    NORMAL = auto()
    CONFIRMANDO = auto()
    CAIDA_CONFIRMADA = auto()


class Posture(Enum):
    """Clasificación postural geométrica, independiente del disparador del Transformer."""
    STANDING = auto()
    GROUND = auto()
    SEATED_STABLE = auto()
    AMBIGUOUS = auto()


@dataclass
class NavigationTelemetry:
    """
    Estima la orientación relativa y distancia aproximada de la persona.

    La distancia es una aproximación monocular y se actualiza solamente cuando
    la persona se encuentra en una postura compatible con bipedestación.
    """
    target_center_px: Tuple[int, int]
    bearing_deg: float           # Azimut respecto al eje óptico (proyección pinhole). + = derecha
    estimated_distance_m: float  # Distancia aproximada objetivo-cámara (m); válida sobre todo de pie
    is_target_valid: bool


@dataclass
class DetectionResult:
    """
    Resultado generado por FallDetector para un frame procesado.

    Agrupa el estado de detección, métricas posturales, presencia de la
    persona, telemetría de navegación y landmarks estimados.
    """
    state: FallState
    posture: Posture
    fall_confidence: float
    raw_fall_detected: bool
    torso_angle: float
    aspect_ratio: float
    is_grounded: bool
    is_standing: bool
    time_in_fall: float
    person_detected: bool
    telemetry: NavigationTelemetry
    pose_landmarks: Optional[landmark_pb2.NormalizedLandmarkList] = None


class FallDetector:
    """
    Integra inferencia TFLite, análisis geométrico y una FSM para detectar caídas.

    Cada frame produce un DetectionResult que contiene el estado del detector,
    métricas posturales y telemetría visual para el futuro subsistema de movilidad.
    """
    def __init__(
        self,
        model_path: str = "models/fall_detection_transformer.tflite",
        config: Optional[DetectorConfig] = None,
    ) -> None:
        self.cfg = config or DetectorConfig()

        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {resolved_path}")

        with open(resolved_path, "rb") as f:
            model_bytes = f.read()

        self.interpreter = Interpreter(model_content=model_bytes)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.feature_sequence: deque = deque(maxlen=self.cfg.feature_window_size)
        self.state = FallState.NORMAL
        self.fall_start_time: Optional[float] = None
        self.recovery_start_time: Optional[float] = None
        self.ambiguous_since: Optional[float] = None
        self.impact_confidence: float = 0.0

        # Memoria biomecánica adaptativa (altura de referencia erguida, en píxeles).
        # No se reinicia en reset(): es una calibración de sesión, no un estado de la FSM.
        self.standing_height_px: Optional[float] = None

        # Filtros pasa-bajos (EMA); todos comparten el alpha configurado centralmente.
        self.ema_torso_angle = ExponentialMovingAverage(self.cfg.smoothing_alpha)
        self.ema_aspect_ratio = ExponentialMovingAverage(self.cfg.smoothing_alpha)
        self.ema_bearing = ExponentialMovingAverage(self.cfg.smoothing_alpha)
        self.ema_distance = ExponentialMovingAverage(self.cfg.smoothing_alpha)

    # ------------------------------------------------------------------ #
    # Normalización de esqueleto (entrada del Transformer)
    # ------------------------------------------------------------------ #
    def _normalize_skeleton_punpayut(self, frame_features_sorted: np.ndarray) -> np.ndarray:
        normalized = np.copy(frame_features_sorted)
        min_conf = self.cfg.min_keypoint_confidence

        ls_i, rs_i = KPT_INDEX["Left Shoulder"] * 3, KPT_INDEX["Right Shoulder"] * 3
        lh_i, rh_i = KPT_INDEX["Left Hip"] * 3, KPT_INDEX["Right Hip"] * 3

        ls_x, ls_y, ls_c = frame_features_sorted[ls_i:ls_i + 3]
        rs_x, rs_y, rs_c = frame_features_sorted[rs_i:rs_i + 3]
        lh_x, lh_y, lh_c = frame_features_sorted[lh_i:lh_i + 3]
        rh_x, rh_y, rh_c = frame_features_sorted[rh_i:rh_i + 3]

        mid_sh_x, mid_sh_y, _ = _robust_midpoint(ls_x, ls_y, ls_c, rs_x, rs_y, rs_c, min_conf)
        mid_hip_x, mid_hip_y, hip_valid = _robust_midpoint(lh_x, lh_y, lh_c, rh_x, rh_y, rh_c, min_conf)

        if not hip_valid:
            return frame_features_sorted

        ref_height = abs(mid_sh_y - mid_hip_y) if not math.isnan(mid_sh_y) else math.nan
        perform_scaling = not (math.isnan(ref_height) or ref_height < 1e-5)

        for name in KEYPOINT_NAMES:
            idx = KPT_INDEX[name] * 3
            normalized[idx] -= mid_hip_x
            normalized[idx + 1] -= mid_hip_y
            if perform_scaling:
                normalized[idx] /= ref_height
                normalized[idx + 1] /= ref_height

        return normalized

    # ------------------------------------------------------------------ #
    # Geometría postural
    # ------------------------------------------------------------------ #
    def _extract_bbox(self, lms, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """Caja delimitadora (ancho_px, alto_px) de los keypoints visibles."""
        min_conf = self.cfg.min_keypoint_confidence
        valid_x = [lm.x * frame_w for lm in lms if lm.visibility > min_conf]
        valid_y = [lm.y * frame_h for lm in lms if lm.visibility > min_conf]
        if len(valid_x) < 4 or len(valid_y) < 4:
            return 0.0, 0.0
        return max(valid_x) - min(valid_x), max(valid_y) - min(valid_y)

    def _torso_geometry(
            self, ls, rs, lh, rh, nose,
        ) -> Tuple[Optional[float], bool, float, float, bool]:
            """
            Calcula el ángulo del torso combinando plano 2D y eje Z para caídas sagitales.
            """
            min_conf = self.cfg.min_keypoint_confidence
            sh_x, sh_y, sh_valid = _robust_midpoint(ls.x, ls.y, ls.visibility, rs.x, rs.y, rs.visibility, min_conf)
            hip_x, hip_y, hip_valid = _robust_midpoint(lh.x, lh.y, lh.visibility, rh.x, rh.y, rh.visibility, min_conf)

            if sh_valid and hip_valid:
                sh_z = (ls.z + rs.z) / 2.0
                hip_z = (lh.z + rh.z) / 2.0

                dx = sh_x - hip_x
                dy = sh_y - hip_y
                dz = sh_z - hip_z

                # Ángulo 2D (caídas laterales) vs 3D (caídas sagitales hacia/desde la cámara)
                torso_angle_2d = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))
                h_dist_3d = math.sqrt(dx * dx + dz * dz)
                torso_angle_3d = math.degrees(math.atan2(abs(dy), h_dist_3d + 1e-6))
                torso_angle = min(torso_angle_2d, torso_angle_3d)

                # De pie únicamente si los hombros están sobre las caderas Y el torso está erguido
                shoulders_above_hips = sh_y < hip_y
                is_standing = shoulders_above_hips and (torso_angle_2d > self.cfg.upright_torso_angle)

                centroid_x = (sh_x + hip_x) / 2.0
                centroid_y = (sh_y + hip_y) / 2.0
                return torso_angle, is_standing, centroid_x, centroid_y, True

            if sh_valid:
                is_standing = (sh_y < self.cfg.shoulder_high_position_ratio and nose.visibility > min_conf)
                return None, is_standing, sh_x, sh_y, True

            return None, False, math.nan, math.nan, False

    def _classify_posture(
        self, effective_angle: float, effective_ratio: float,
        bbox_height_px: float, is_standing: bool,
    ) -> Posture:
        """
        Jerarquía estricta: La evidencia física de suelo SIEMPRE tiene prioridad
        sobre cualquier heurística de bipedestación.
        """
        height_ratio: Optional[float] = None
        if self.standing_height_px and bbox_height_px > 0:
            height_ratio = bbox_height_px / self.standing_height_px

        torso_down = effective_angle <= self.cfg.torso_angle_threshold
        box_horizontal = effective_ratio >= self.cfg.aspect_ratio_threshold
        height_collapsed = height_ratio is not None and height_ratio <= self.cfg.height_collapse_ratio

        # 1. Prioridad Máxima: Si el cuerpo está horizontal o colapsado, es SUELO
        if torso_down or box_horizontal or height_collapsed:
            return Posture.GROUND

        # 2. Si no está en el suelo, evaluar si está erguido de pie
        if is_standing:
            return Posture.STANDING

        # 3. Postura sentada intermedia
        is_seated_band = (
            height_ratio is not None
            and self.cfg.height_collapse_ratio < height_ratio <= self.cfg.seated_height_ratio_max
        )
        if is_seated_band and effective_angle > self.cfg.torso_angle_threshold:
            return Posture.SEATED_STABLE

        return Posture.AMBIGUOUS

    # ------------------------------------------------------------------ #
    # Telemetría de navegación (guiado de la base móvil)
    # ------------------------------------------------------------------ #
    def _navigation_telemetry(
        self, centroid_x: float, centroid_y: float, is_valid: bool,
        bbox_height_px: float, is_standing: bool, frame_w: int, frame_h: int,
    ) -> NavigationTelemetry:
        if not is_valid:
            return NavigationTelemetry(
                target_center_px=(frame_w // 2, frame_h // 2),
                bearing_deg=self.ema_bearing.value or 0.0,
                estimated_distance_m=self.ema_distance.value or 0.0,
                is_target_valid=False,
            )

        target_px = (int(centroid_x * frame_w), int(centroid_y * frame_h))

        # Proyección pinhole real: foco en píxeles derivado del HFOV de la cámara,
        # en vez de escalar linealmente el error normalizado por el HFOV completo
        # (esa aproximación diverge de la tangente real hacia los bordes del cuadro).
        focal_px = frame_w / (2.0 * math.tan(math.radians(self.cfg.camera_hfov_deg / 2.0)))
        pixel_offset_x = target_px[0] - (frame_w / 2.0)
        bearing_deg = math.degrees(math.atan2(pixel_offset_x, focal_px))
        smoothed_bearing = self.ema_bearing.update(bearing_deg)

        # La distancia por altura de bbox solo es válida con la persona de pie: en
        # el suelo la bbox deja de representar la altura corporal real. En esos
        # casos se conserva la última muestra válida en vez de emitir un valor
        # físicamente absurdo.
        if is_standing and bbox_height_px > 1.0:
            raw_distance_m = (self.cfg.assumed_person_height_m * focal_px) / bbox_height_px
            smoothed_distance = self.ema_distance.update(raw_distance_m)
        else:
            smoothed_distance = self.ema_distance.value if self.ema_distance.value is not None else 0.0

        return NavigationTelemetry(
            target_center_px=target_px,
            bearing_deg=smoothed_bearing,
            estimated_distance_m=smoothed_distance,
            is_target_valid=True,
        )

    # ------------------------------------------------------------------ #
    # Orquestación de métricas por frame
    # ------------------------------------------------------------------ #
    def _compute_metrics(
        self, landmarks, frame_w: int, frame_h: int,
    ) -> Tuple[float, float, Posture, NavigationTelemetry]:
        if not landmarks:
            held_angle = self.ema_torso_angle.value if self.ema_torso_angle.value is not None else 90.0
            held_ratio = self.ema_aspect_ratio.value if self.ema_aspect_ratio.value is not None else 0.0
            empty_nav = self._navigation_telemetry(math.nan, math.nan, False, 0.0, False, frame_w, frame_h)
            return held_angle, held_ratio, Posture.AMBIGUOUS, empty_nav

        lms = landmarks.landmark
        ls = lms[_POSE_LANDMARK.LEFT_SHOULDER.value]
        rs = lms[_POSE_LANDMARK.RIGHT_SHOULDER.value]
        lh = lms[_POSE_LANDMARK.LEFT_HIP.value]
        rh = lms[_POSE_LANDMARK.RIGHT_HIP.value]
        nose = lms[_POSE_LANDMARK.NOSE.value]

        bbox_w_px, bbox_h_px = self._extract_bbox(lms, frame_w, frame_h)
        aspect_ratio_raw = bbox_w_px / max(bbox_h_px, 1.0)

        torso_angle_raw, is_standing, centroid_x, centroid_y, geometry_valid = self._torso_geometry(
            ls, rs, lh, rh, nose
        )

        effective_ratio = self.ema_aspect_ratio.update(aspect_ratio_raw)
        if torso_angle_raw is not None:
            effective_angle = self.ema_torso_angle.update(torso_angle_raw)
        else:
            effective_angle = self.ema_torso_angle.value if self.ema_torso_angle.value is not None else 90.0

        # Calibración adaptativa de altura de referencia (solo mientras está de pie).
        if is_standing and bbox_h_px > 120:
            self.standing_height_px = (
                bbox_h_px if self.standing_height_px is None else max(self.standing_height_px, bbox_h_px)
            )

        posture = self._classify_posture(effective_angle, effective_ratio, bbox_h_px, is_standing)
        telemetry = self._navigation_telemetry(
            centroid_x, centroid_y, geometry_valid, bbox_h_px, is_standing, frame_w, frame_h
        )

        return effective_angle, effective_ratio, posture, telemetry

    # ------------------------------------------------------------------ #
    # FSM
    # ------------------------------------------------------------------ #
    def _update_fsm(
        self, raw_fall_detected: bool, posture: Posture, person_detected: bool, now: float, prob_caida: float
    ) -> float:
        """
        Diagrama de transiciones:

        NORMAL --(raw_fall_detected)--> CONFIRMANDO
        CONFIRMANDO --(GROUND sostenido >= confirmation_time_sec)--> CAIDA_CONFIRMADA
        CONFIRMANDO --(STANDING)--> NORMAL
        CONFIRMANDO --(ni GROUND ni STANDING durante >= ambiguous_timeout_sec)--> NORMAL
            Escape de deadlock: una postura AMBIGUOUS o SEATED_STABLE sostenida
            no es evidencia suficiente de una caída real en curso (p. ej. la
            persona se sentó por su cuenta antes del umbral de confirmación).
        CAIDA_CONFIRMADA --(STANDING o SEATED_STABLE sostenidos >= recovery_hold_sec)--> NORMAL
            Soporta que la persona quede sentada (asistida por un cuidador o por
            sí misma) sin exigir que vuelva a ponerse de pie para desactivar la
            alerta visual.
        """
        if self.cfg.confirmation_time_sec <= 0.0:
            self.state = FallState.CAIDA_CONFIRMADA if raw_fall_detected else FallState.NORMAL
            return 0.0

        time_in_fall = 0.0

        if self.state == FallState.NORMAL:
            if raw_fall_detected:
                self.state = FallState.CONFIRMANDO
                self.fall_start_time = now
                self.ambiguous_since = None
                self.recovery_start_time = None
                self.impact_confidence = prob_caida

        elif self.state == FallState.CONFIRMANDO:
            time_in_fall = now - self.fall_start_time
            self.impact_confidence = max(self.impact_confidence, prob_caida)
            recently_occluded = not person_detected and time_in_fall < self.cfg.post_impact_grace_sec
            on_ground = posture == Posture.GROUND or recently_occluded

            if on_ground:
                self.ambiguous_since = None
                if time_in_fall >= self.cfg.confirmation_time_sec:
                    self.state = FallState.CAIDA_CONFIRMADA
            elif posture == Posture.STANDING:
                self.state = FallState.NORMAL
                self.fall_start_time = None
                self.ambiguous_since = None
                self.impact_confidence = 0.0
            else:
                # Posture.AMBIGUOUS o Posture.SEATED_STABLE
                if self.ambiguous_since is None:
                    self.ambiguous_since = now
                elif (now - self.ambiguous_since) >= self.cfg.ambiguous_timeout_sec:
                    self.state = FallState.NORMAL
                    self.fall_start_time = None
                    self.ambiguous_since = None
                    self.impact_confidence = 0.0

        elif self.state == FallState.CAIDA_CONFIRMADA:
            time_in_fall = now - self.fall_start_time
            stable_recovery = person_detected and posture in (Posture.STANDING, Posture.SEATED_STABLE)
            if stable_recovery:
                if self.recovery_start_time is None:
                    self.recovery_start_time = now
                elif (now - self.recovery_start_time) >= self.cfg.recovery_hold_sec:
                    self.state = FallState.NORMAL
                    self.fall_start_time = None
                    self.recovery_start_time = None
                    self.impact_confidence = 0.0
            else:
                self.recovery_start_time = None

        return time_in_fall

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def process_frame(
        self, frame_bgr: np.ndarray, current_timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """
        Procesa un frame y retorna el estado completo del detector.

        Incluye inferencia temporal, cálculo geométrico, actualización de la FSM
        y generación de telemetría visual.
        """
        if current_timestamp is None:
            current_timestamp = time.monotonic()

        h, w, _ = frame_bgr.shape
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.pose.process(img_rgb)

        frame_features = np.zeros(FEATURE_DIM, dtype=np.float32)
        person_detected = bool(results.pose_landmarks)

        if person_detected:
            for mp_enum, name in _LANDMARK_ENUM_TO_NAME.items():
                lm = results.pose_landmarks.landmark[mp_enum.value]
                idx = KPT_INDEX[name] * 3
                frame_features[idx] = lm.x
                frame_features[idx + 1] = lm.y
                frame_features[idx + 2] = lm.visibility

        norm_features = self._normalize_skeleton_punpayut(frame_features)
        self.feature_sequence.append(norm_features)

        prob_caida = 0.0
        if len(self.feature_sequence) == self.cfg.feature_window_size:
            input_data = np.expand_dims(np.array(self.feature_sequence, dtype=np.float32), axis=0)
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            prob_caida = float(self.interpreter.get_tensor(self.output_details[0]["index"])[0][0])

        raw_fall_detected = prob_caida >= self.cfg.tflite_conf_threshold
        torso_angle, aspect_ratio, posture, telemetry = self._compute_metrics(
            results.pose_landmarks, frame_w=w, frame_h=h
        )

        time_in_fall = self._update_fsm(raw_fall_detected, posture, person_detected, current_timestamp, prob_caida)

        # Si está evaluando o confirmando caída, reportar la certeza del impacto dinámico
        reported_confidence = (
            self.impact_confidence
            if self.state in (FallState.CONFIRMANDO, FallState.CAIDA_CONFIRMADA)
            else prob_caida
        )

        return DetectionResult(
            state=self.state,
            posture=posture,
            fall_confidence=reported_confidence,
            raw_fall_detected=raw_fall_detected,
            torso_angle=torso_angle,
            aspect_ratio=aspect_ratio,
            is_grounded=(posture == Posture.GROUND),
            is_standing=(posture == Posture.STANDING),
            time_in_fall=time_in_fall,
            person_detected=person_detected,
            telemetry=telemetry,
            pose_landmarks=results.pose_landmarks,
        )

    def reset(self) -> None:
        self.feature_sequence.clear()
        self.state = FallState.NORMAL
        self.fall_start_time = None
        self.recovery_start_time = None
        self.ambiguous_since = None
        self.impact_confidence = 0.0
        self.ema_torso_angle.reset()
        self.ema_aspect_ratio.reset()
        self.ema_bearing.reset()
        self.ema_distance.reset()
