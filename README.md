# Robot Móvil Asistente para Detección de Caídas

Proyecto desarrollado como parte del Trabajo de Título de Ingeniería Civil en Informática.

El sistema busca servir como base para un robot móvil asistencial capaz de detectar caídas, generar alertas remotas y permitir solicitudes de ayuda mediante comandos de voz. La versión actual corresponde a la **Etapa 2: Diseño e Implementación Inicial**, por lo que el desarrollo se concentra principalmente en los módulos de software y su evaluación preliminar.

## Funcionalidades actuales

- Estimación de pose mediante MediaPipe Pose.
- Detección temporal de caídas mediante un modelo TFLite basado en el trabajo de [Punpayut](https://github.com/punpayut/Fall-Detection).
- Verificación geométrica de postura.
- Confirmación de eventos mediante una máquina de estados finita (FSM).
- Suavizado temporal de métricas mediante EMA.
- Estimación visual de orientación y distancia para futura navegación.
- Generación de evidencias y alertas mediante Telegram.
- Reconocimiento local de comandos de voz mediante Vosk.
- Comandos de emergencia y cancelación por voz.

## Estructura principal

```text
.
├── main.py
├── src/
│   ├── fall_detector.py
│   ├── detector_config.py
│   ├── alert_manager.py
│   └── voice_manager.py
├── models/
├── tests/
├── evidences/
└── requirements.txt
```

## Instalación

Se recomienda utilizar un entorno virtual de Python.

```bash
python -m venv .venv
```

Activación en Windows:

```bash
.venv\Scripts\activate
```

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

## Modelos requeridos

El proyecto requiere los siguientes modelos dentro del directorio `models/`:

```text
models/
├── fall_detection_transformer.tflite
└── vosk-model-es/
```

El modelo TFLite utilizado corresponde a la arquitectura seleccionada a partir del repositorio de Punpayut.

## Configuración de Telegram

Para habilitar las alertas remotas se debe crear un archivo `.env` en la raíz del proyecto:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

El archivo `.env` no debe incorporarse al repositorio.

## Ejecución

Con webcam:

```bash
python main.py --source 0
```

Con un archivo de video:

```bash
python main.py --source ruta/al/video.mp4
```

Durante la ejecución:

- `Q` o `ESC`: cerrar el programa.
- `R`: reiniciar el estado del detector.

## Comandos de voz

Ejemplos de palabras reconocidas para solicitar asistencia:

```text
ayuda
emergencia
socorro
llamar
auxilio
```

Ejemplos para cancelar una alerta:

```text
estoy bien
falsa alarma
cancelar
detente
me equivoqué
```

## Estado actual

La versión correspondiente a la Etapa 2 ha sido evaluada principalmente en un entorno de desarrollo sobre PC y videos previamente registrados.

Como resultados preliminares:

- se detectaron 46 de 50 videos de caída en una auditoría de sensibilidad;
- se eliminaron 43 de 103 falsos positivos identificados previamente.

Estas pruebas no representan todavía la validación final del prototipo.

La integración con Raspberry Pi 5, motores, sensores de obstáculos y plataforma móvil física queda pendiente para las siguientes etapas del proyecto.