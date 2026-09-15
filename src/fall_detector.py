"""
Módulo de Detección de Caídas para Robot Móvil Asistencial.
Arquitectura Híbrida v3:
- Cinemática: Red Transformer TFLite (disparador dinámico).
- Geometría Relativa: Ángulo 3D del tronco y Relación de Aspecto (W/H).
- Velocidad de descenso: Derivada vertical del centroide (dY/dt).
- FSM con Histéresis: Evita parpadeos y asegura persistencia.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path
import time
from typing import Optional, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

Interpreter = tf.lite.Interpreter


class FallState(Enum):
    NORMAL = auto()
    CONFIRMANDO = auto()
    CAIDA_CONFIRMADA = auto()


@dataclass
class DetectionResult:
    state: FallState
    fall_confidence: float
    raw_fall_detected: bool
    torso_angle: float
    aspect_ratio: float
    vertical_velocity: float
    is_grounded: bool
    keypoints: np.ndarray
    time_in_fall: float
    person_detected: bool
    pose_landmarks: Optional[Any] = None


class FallDetector:
    def __init__(
        self,
        model_path: str = "models/fall_detection_transformer.tflite",
        fall_confidence_threshold: float = 0.80,
        confirmation_time_sec: float = 1.2,
        torso_angle_threshold: float = 50.0,
        aspect_ratio_threshold: float = 1.0,
        min_keypoint_conf: float = 0.30
    ):
        self.conf_threshold = fall_confidence_threshold
        self.confirmation_time_sec = confirmation_time_sec
        self.torso_angle_thresh = torso_angle_threshold
        self.aspect_ratio_thresh = aspect_ratio_threshold
        self.min_keypoint_conf = min_keypoint_conf

        # 1. Carga TFLite
        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {resolved_path}")

        with open(resolved_path, "rb") as f:
            model_bytes = f.read()

        self.interpreter = Interpreter(model_content=model_bytes)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 2. MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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

        self.feature_sequence = deque(maxlen=30)
        self.state = FallState.NORMAL
        self.fall_start_time: Optional[float] = None
        self.recovery_start_time: Optional[float] = None

        # Historial para cálculo de velocidad vertical del centro de masa
        self.prev_hip_y: Optional[float] = None
        self.prev_time: Optional[float] = None

    def _normalize_skeleton_punpayut(self, frame_features_sorted: np.ndarray) -> np.ndarray:
        normalized = np.copy(frame_features_sorted)

        ls_i = self.kpt_dict['Left Shoulder'] * 3
        rs_i = self.kpt_dict['Right Shoulder'] * 3
        lh_i = self.kpt_dict['Left Hip'] * 3
        rh_i = self.kpt_dict['Right Hip'] * 3

        ls_x, ls_y, ls_c = frame_features_sorted[ls_i:ls_i + 3]
        rs_x, rs_y, rs_c = frame_features_sorted[rs_i:rs_i + 3]
        lh_x, lh_y, lh_c = frame_features_sorted[lh_i:lh_i + 3]
        rh_x, rh_y, rh_c = frame_features_sorted[rh_i:rh_i + 3]

        mid_sh_x, mid_sh_y = np.nan, np.nan
        val_ls, val_rs = ls_c > self.min_keypoint_conf, rs_c > self.min_keypoint_conf
        if val_ls and val_rs:
            mid_sh_x, mid_sh_y = (ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0
        elif val_ls:
            mid_sh_x, mid_sh_y = ls_x, ls_y
        elif val_rs:
            mid_sh_x, mid_sh_y = rs_x, rs_y

        mid_hip_x, mid_hip_y = np.nan, np.nan
        val_lh, val_rh = lh_c > self.min_keypoint_conf, rh_c > self.min_keypoint_conf
        if val_lh and val_rh:
            mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0
        elif val_lh:
            mid_hip_x, mid_hip_y = lh_x, lh_y
        elif val_rh:
            mid_hip_x, mid_hip_y = rh_x, rh_y

        if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
            return frame_features_sorted

        ref_height = np.nan
        if not np.isnan(mid_sh_y) and not np.isnan(mid_hip_y):
            ref_height = np.abs(mid_sh_y - mid_hip_y)

        perform_scaling = not (np.isnan(ref_height) or ref_height < 1e-5)

        for name in self.sorted_names:
            idx = self.kpt_dict[name] * 3
            normalized[idx] -= mid_hip_x
            normalized[idx + 1] -= mid_hip_y
            if perform_scaling:
                normalized[idx] /= ref_height
                normalized[idx + 1] /= ref_height

        return normalized

    def _compute_metrics(self, landmarks, current_timestamp: float) -> Tuple[float, float, float, bool]:
        """
        Calcula métricas corporales puramente relativas e invariantes al tamaño del encuadre.
        """
        if not landmarks:
            return 90.0, 0.0, 0.0, False

        lms = landmarks.landmark
        ls = lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lms[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = lms[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lms[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 1. Ángulo 3D del tronco (hombros a caderas)
        torso_angle = 90.0
        current_hip_y = None

        if (ls.visibility > self.min_keypoint_conf or rs.visibility > self.min_keypoint_conf) and \
           (lh.visibility > self.min_keypoint_conf or rh.visibility > self.min_keypoint_conf):
            sh_x = (ls.x + rs.x) / 2.0
            sh_y = (ls.y + rs.y) / 2.0
            sh_z = (ls.z + rs.z) / 2.0

            hip_x = (lh.x + rh.x) / 2.0
            hip_y = (lh.y + rh.y) / 2.0
            hip_z = (lh.z + rh.z) / 2.0
            current_hip_y = hip_y

            dx = sh_x - hip_x
            dy = sh_y - hip_y
            dz = sh_z - hip_z

            h_dist = math.sqrt(dx * dx + dz * dz)
            v_dist = abs(dy)
            torso_angle = math.degrees(math.atan2(v_dist, h_dist + 1e-6))

        # 2. Relación de aspecto W/H usando puntos normalizados visibles
        valid_x = [lm.x for lm in lms if lm.visibility > self.min_keypoint_conf]
        valid_y = [lm.y for lm in lms if lm.visibility > self.min_keypoint_conf]
        aspect_ratio = 0.0
        if len(valid_x) >= 4:
            w = max(valid_x) - min(valid_x)
            h = max(valid_y) - min(valid_y)
            aspect_ratio = float(w / max(h, 1e-4))

        # 3. Velocidad vertical de caderas (dY/dt)
        vertical_vel = 0.0
        if current_hip_y is not None and self.prev_hip_y is not None and self.prev_time is not None:
            dt = current_timestamp - self.prev_time
            if dt > 1e-4:
                # dy positivo significa movimiento hacia abajo en coordenadas normalizadas
                vertical_vel = (current_hip_y - self.prev_hip_y) / dt

        if current_hip_y is not None:
            self.prev_hip_y = current_hip_y
            self.prev_time = current_timestamp

        # 4. Condición de permanencia en suelo (puramente geométrica)
        is_grounded = (torso_angle <= self.torso_angle_thresh) or (aspect_ratio >= self.aspect_ratio_thresh)

        return torso_angle, aspect_ratio, vertical_vel, is_grounded

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        current_timestamp: Optional[float] = None
    ) -> DetectionResult:
        if current_timestamp is None:
            current_timestamp = time.monotonic()

        h, w, _ = frame_bgr.shape
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.pose.process(img_rgb)

        frame_features = np.zeros(51, dtype=np.float32)
        raw_keypoints = np.zeros((17, 3), dtype=np.float32)
        person_detected = False

        if results.pose_landmarks:
            person_detected = True
            for mp_enum, name in self.mapping.items():
                lm = results.pose_landmarks.landmark[mp_enum.value]
                idx = self.kpt_dict[name] * 3
                frame_features[idx] = lm.x
                frame_features[idx + 1] = lm.y
                frame_features[idx + 2] = lm.visibility
                raw_keypoints[self.kpt_dict[name]] = [lm.x * w, lm.y * h, lm.visibility]

        norm_features = self._normalize_skeleton_punpayut(frame_features)
        self.feature_sequence.append(norm_features)

        prob_caida = 0.0
        if len(self.feature_sequence) == 30:
            input_data = np.expand_dims(np.array(self.feature_sequence, dtype=np.float32), axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            prob_caida = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])

        raw_fall_detected = prob_caida >= self.conf_threshold
        torso_angle, aspect_ratio, vertical_vel, is_grounded = self._compute_metrics(
            results.pose_landmarks, current_timestamp
        )

        time_in_fall = 0.0

        # --- FSM con Histéresis Robusta ---
        if self.confirmation_time_sec <= 0.0:
            self.state = FallState.CAIDA_CONFIRMADA if raw_fall_detected else FallState.NORMAL
        else:
            if self.state == FallState.NORMAL:
                # Gatillo: disparo de red neuronal con velocidad hacia abajo o torso inclinado
                if raw_fall_detected:
                    self.state = FallState.CONFIRMANDO
                    self.fall_start_time = current_timestamp
                    self.recovery_start_time = None

            elif self.state == FallState.CONFIRMANDO:
                time_in_fall = current_timestamp - self.fall_start_time

                # Si el cuerpo está en el suelo o hubo oclusión temporal tras el impacto
                if is_grounded or (not person_detected and time_in_fall < 1.0):
                    if time_in_fall >= self.confirmation_time_sec:
                        self.state = FallState.CAIDA_CONFIRMADA
                else:
                    # Cancelación limpia: torso erguido (>65°) y relación de aspecto vertical
                    if torso_angle > 65.0 and aspect_ratio < 0.8:
                        self.state = FallState.NORMAL
                        self.fall_start_time = None

            elif self.state == FallState.CAIDA_CONFIRMADA:
                time_in_fall = current_timestamp - self.fall_start_time

                # Recuperación garantizada: persona de pie erguida (>65°) durante 1.0s continuo
                if torso_angle > 65.0 and aspect_ratio < 0.85:
                    if self.recovery_start_time is None:
                        self.recovery_start_time = current_timestamp
                    elif current_timestamp - self.recovery_start_time >= 1.0:
                        self.state = FallState.NORMAL
                        self.fall_start_time = None
                        self.recovery_start_time = None
                else:
                    self.recovery_start_time = None

        return DetectionResult(
            state=self.state,
            fall_confidence=prob_caida,
            raw_fall_detected=raw_fall_detected,
            torso_angle=torso_angle,
            aspect_ratio=aspect_ratio,
            vertical_velocity=vertical_vel,
            is_grounded=is_grounded,
            keypoints=raw_keypoints,
            time_in_fall=time_in_fall,
            person_detected=person_detected,
            pose_landmarks=results.pose_landmarks
        )

    def reset(self) -> None:
        self.feature_sequence.clear()
        self.state = FallState.NORMAL
        self.fall_start_time = None
        self.recovery_start_time = None
        self.prev_hip_y = None
        self.prev_time = None