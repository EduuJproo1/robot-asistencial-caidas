"""
Módulo de Detección de Caídas para Robot Móvil Asistencial.
Integra estimación de postura con MediaPipe, clasificación temporal con TFLite
y máquina de confirmación temporal para mitigación de falsas alarmas (RF3).
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

Interpreter = tf.lite.Interpreter

class FallState(Enum):
    NORMAL = auto()             # Monitoreo estándar; persona erguida o caminando
    CONFIRMANDO = auto()        # Posible caída detectada; evaluando persistencia
    CAIDA_CONFIRMADA = auto()   # Caída persistente en el suelo; condición de alerta


@dataclass
class DetectionResult:
    """Estructura estandarizada con la telemetría de salida de cada fotograma."""
    state: FallState
    fall_confidence: float
    keypoints: np.ndarray        # Forma (17, 3) en píxeles (x, y, visibilidad)
    time_in_fall: float          # Tiempo acumulado en condición de caída (segundos)
    person_detected: bool        # Flag que indica si hay sujeto visible en encuadre


class FallDetector:
    def __init__(
        self,
        model_path: str = "models/fall_detection_transformer.tflite",
        fall_confidence_threshold: float = 0.90,
        confirmation_time_sec: float = 1.5
    ):
        self.conf_threshold = fall_confidence_threshold
        self.confirmation_time_sec = confirmation_time_sec

        # 1. Carga del modelo TFLite en memoria (evita errores con tildes en Windows)
        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo TFLite en: {resolved_path}")

        with open(resolved_path, "rb") as f:
            model_bytes = f.read()

        self.interpreter = Interpreter(model_content=model_bytes)
        self.interpreter.allocate_tensors()


        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 2. Configuración de MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Mapeo de keypoints COCO
        self.mapping = {
            self.mp_pose.PoseLandmark.NOSE: 'Nose',
            self.mp_pose.PoseLandmark.LEFT_EYE: 'Left Eye',
            self.mp_pose.PoseLandmark.RIGHT_EYE: 'Right Eye',
            self.mp_pose.PoseLandmark.LEFT_EAR: 'Left Ear',
            self.mp_pose.PoseLandmark.RIGHT_EAR: 'Right Ear',
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder',
            self.mp_pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
            self.mp_pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow',
            self.mp_pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
            self.mp_pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist',
            self.mp_pose.PoseLandmark.LEFT_HIP: 'Left Hip',
            self.mp_pose.PoseLandmark.RIGHT_HIP: 'Right Hip',
            self.mp_pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
            self.mp_pose.PoseLandmark.RIGHT_KNEE: 'Right Knee',
            self.mp_pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
            self.mp_pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle'
        }
        self.sorted_names = sorted(list(self.mapping.values()))
        self.kpt_dict = {name: i for i, name in enumerate(self.sorted_names)}

        # Buffer temporal de 30 fotogramas
        self.feature_sequence = deque(maxlen=30)

        # Variables internas de la máquina de estados
        self.state = FallState.NORMAL
        self.fall_start_time: Optional[float] = None

    def _normalize_skeleton(self, features: np.ndarray) -> np.ndarray:
        norm = np.copy(features)

        ls_i, rs_i = self.kpt_dict['Left Shoulder'] * 3, self.kpt_dict['Right Shoulder'] * 3
        lh_i, rh_i = self.kpt_dict['Left Hip'] * 3, self.kpt_dict['Right Hip'] * 3

        lh_v = features[lh_i + 2] > 0.3
        rh_v = features[rh_i + 2] > 0.3

        if lh_v and rh_v:
            mid_hip_x = (features[lh_i] + features[rh_i]) / 2.0
            mid_hip_y = (features[lh_i + 1] + features[rh_i + 1]) / 2.0
        elif lh_v:
            mid_hip_x, mid_hip_y = features[lh_i], features[lh_i + 1]
        elif rh_v:
            mid_hip_x, mid_hip_y = features[rh_i], features[rh_i + 1]
        else:
            return features

        ls_v = features[ls_i + 2] > 0.3
        rs_v = features[rs_i + 2] > 0.3
        mid_sh_y = np.nan

        if ls_v and rs_v:
            mid_sh_y = (features[ls_i + 1] + features[rs_i + 1]) / 2.0
        elif ls_v:
            mid_sh_y = features[ls_i + 1]
        elif rs_v:
            mid_sh_y = features[rs_i + 1]

        scale = np.abs(mid_sh_y - mid_hip_y) if not np.isnan(mid_sh_y) else np.nan
        do_scale = not (np.isnan(scale) or scale < 1e-5)

        for name in self.sorted_names:
            idx = self.kpt_dict[name] * 3
            norm[idx] -= mid_hip_x
            norm[idx + 1] -= mid_hip_y
            if do_scale:
                norm[idx] /= scale
                norm[idx + 1] /= scale

        return norm

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        current_timestamp: Optional[float] = None
    ) -> DetectionResult:
        """
        Procesa un único fotograma y actualiza la máquina de estados.
        """
        if current_timestamp is None:
            current_timestamp = time.monotonic()

        h, w, _ = frame_bgr.shape
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.pose.process(img_rgb)

        raw_features = np.zeros(51, dtype=np.float32)
        raw_keypoints = np.zeros((17, 3), dtype=np.float32)
        person_detected = False

        if results.pose_landmarks:
            person_detected = True
            for enum_kpt, name in self.mapping.items():
                lm = results.pose_landmarks.landmark[enum_kpt.value]
                idx = self.kpt_dict[name] * 3
                raw_features[idx] = lm.x
                raw_features[idx + 1] = lm.y
                raw_features[idx + 2] = lm.visibility

                # Coordenadas en píxeles para el módulo de seguimiento de la persona
                raw_keypoints[self.kpt_dict[name]] = [lm.x * w, lm.y * h, lm.visibility]

        norm_features = self._normalize_skeleton(raw_features)
        self.feature_sequence.append(norm_features)

        prob_caida = 0.0
        if len(self.feature_sequence) == 30:
            input_data = np.expand_dims(np.array(self.feature_sequence, dtype=np.float32), axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            prob_caida = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])

        # --- Máquina de Estados Finita (RF3) ---
        time_in_fall = 0.0
        is_fall_candidate = prob_caida >= self.conf_threshold

        if is_fall_candidate:
            if self.state == FallState.NORMAL:
                self.state = FallState.CONFIRMANDO
                self.fall_start_time = current_timestamp

            elif self.state == FallState.CONFIRMANDO:
                time_in_fall = current_timestamp - self.fall_start_time
                if time_in_fall >= self.confirmation_time_sec:
                    self.state = FallState.CAIDA_CONFIRMADA

            elif self.state == FallState.CAIDA_CONFIRMADA:
                time_in_fall = current_timestamp - self.fall_start_time
        else:
            # Si el sujeto se recupera o fue un movimiento rápido de flexión/sentarse
            if self.state == FallState.CONFIRMANDO:
                self.state = FallState.NORMAL
                self.fall_start_time = None
            elif self.state == FallState.CAIDA_CONFIRMADA:
                # Regreso a estado normal cuando el sujeto se reincorpora
                self.state = FallState.NORMAL
                self.fall_start_time = None

        return DetectionResult(
            state=self.state,
            fall_confidence=prob_caida,
            keypoints=raw_keypoints,
            time_in_fall=time_in_fall,
            person_detected=person_detected
        )

    def reset(self) -> None:
        """Reinicia los buffers y la máquina de estados."""
        self.feature_sequence.clear()
        self.state = FallState.NORMAL
        self.fall_start_time = None