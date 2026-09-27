import argparse
import time
from pathlib import Path

import cv2


parser = argparse.ArgumentParser()
parser.add_argument(
    "categoria",
    choices=["piedra", "papel", "tijera"],
)
parser.add_argument(
    "grupo",
    choices=["entrenamiento", "validacion", "prueba"],
)
parser.add_argument(
    "cantidad",
    type=int,
)
argumentos = parser.parse_args()

categoria = argumentos.categoria
grupo = argumentos.grupo
cantidad = argumentos.cantidad

carpeta = Path("datos") / grupo / categoria
carpeta.mkdir(parents=True, exist_ok=True)

camara = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not camara.isOpened():
    raise RuntimeError("No se ha podido abrir la cámara")

print("Coloca la mano y pulsa ESPACIO para comenzar.")
print("Pulsa Q para cancelar.")

# Esperamos hasta que pulses la barra espaciadora
while True:
    correcto, imagen = camara.read()

    if not correcto:
        raise RuntimeError("No se ha podido obtener la imagen")

    imagen = cv2.flip(imagen, 1)

    cv2.putText(
        imagen,
        f"Preparado para: {categoria}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        imagen,
        "ESPACIO: comenzar | Q: salir",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Captura de imágenes", imagen)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 32:
        break

    if tecla == ord("q"):
        camara.release()
        cv2.destroyAllWindows()
        raise SystemExit

# Capturamos las fotografías
for numero in range(cantidad):
    correcto, imagen = camara.read()

    if not correcto:
        break

    imagen = cv2.flip(imagen, 1)

    archivo = carpeta / f"{categoria}_{time.time_ns()}.jpg"
    cv2.imwrite(str(archivo), imagen)

    cv2.putText(
        imagen,
        f"{categoria}: {numero + 1}/{cantidad}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Captura de imágenes", imagen)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()

print(f"Imágenes guardadas en: {carpeta}")