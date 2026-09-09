"""Prueba de inicialización rápida del detector."""
from src.fall_detector import FallDetector, FallState
import numpy as np

def main():
    print("Iniciando verificación del detector...")
    try:
        detector = FallDetector(model_path="models/fall_detection_transformer.tflite")
        print("✅ Intérprete TFLite y MediaPipe cargados exitosamente.")

        # Fotograma sintético vacío (640x480)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.process_frame(dummy_frame)

        print(f"✅ Inferencia sintética completada. Estado: {res.state.name} | Confianza: {res.fall_confidence:.2f}")
        print("Todo el entorno está operativo.")
    except Exception as e:
        print(f"❌ Error durante la verificación: {e}")

if __name__ == "__main__":
    main()