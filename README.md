# Clasificador de piedra, papel o tijera

Proyecto de visión artificial que clasifica gestos de la mano en tres categorías:

- Papel
- Piedra
- Tijera

El proyecto utiliza transfer learning con MobileNetV2, descargado desde Hugging Face y preentrenado con ImageNet.

## Resultados

- Precisión de validación: 95 %
- Precisión de prueba: 66,67 %
- Ejecución local utilizando la GPU MPS de Apple Silicon

## Tecnologías

- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- OpenCV
- MobileNetV2

## Estructura

- `capturar_imagenes.py`: captura las imágenes mediante la webcam.
- `entrenar.py`: adapta MobileNetV2 a las tres categorías.
- `predecir_camara.py`: realiza predicciones en tiempo real.
- `probar_camara.py`: comprueba el funcionamiento de la cámara.
- `modelo_ppt/`: modelo final entrenado.

Las fotografías utilizadas para entrenar no están incluidas por privacidad.

## Instalación

Se recomienda utilizar Python 3.11 dentro de un entorno de Conda.

```bash
python -m pip install torch torchvision transformers accelerate opencv-python