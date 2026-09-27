import cv2
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


CARPETA_MODELO = "modelo_ppt"
CATEGORIAS = ["papel", "piedra", "tijera"]


# 1. Cargar el modelo que hemos entrenado
procesador = AutoImageProcessor.from_pretrained(
    CARPETA_MODELO
)

modelo = AutoModelForImageClassification.from_pretrained(
    CARPETA_MODELO
)

dispositivo = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

modelo.to(dispositivo)
modelo.eval()


# 2. Aplicar exactamente el mismo preprocesamiento
transformacion = transforms.Compose(
    [
        transforms.CenterCrop((1000, 1000)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=procesador.image_mean,
            std=procesador.image_std,
        ),
    ]
)


# 3. Abrir la cámara integrada del Mac
camara = cv2.VideoCapture(
    1,
    cv2.CAP_AVFOUNDATION,
)

if not camara.isOpened():
    raise RuntimeError("No se ha podido abrir la cámara")

print("Pulsa Q para cerrar.")


while True:
    correcto, imagen = camara.read()

    if not correcto:
        break

    imagen = cv2.flip(imagen, 1)

    # Convertir la imagen de OpenCV al formato utilizado por PyTorch
    imagen_rgb = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2RGB,
    )

    imagen_pil = Image.fromarray(imagen_rgb)

    entrada = transformacion(imagen_pil)
    entrada = entrada.unsqueeze(0).to(dispositivo)

    # Realizar la predicción
    with torch.inference_mode():
        salida = modelo(pixel_values=entrada)
        probabilidades = torch.softmax(
            salida.logits,
            dim=1,
        )

    confianza, indice = probabilidades.max(dim=1)

    categoria = CATEGORIAS[indice.item()]
    porcentaje = confianza.item() * 100

    texto = f"{categoria}: {porcentaje:.1f}%"

    # Dibujar la zona central que analiza el modelo
    alto, ancho = imagen.shape[:2]
    lado = min(1000, alto, ancho)

    x1 = (ancho - lado) // 2
    y1 = (alto - lado) // 2
    x2 = x1 + lado
    y2 = y1 + lado

    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    cv2.putText(
        imagen,
        texto,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    cv2.imshow(
        "Piedra, papel o tijera",
        imagen,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camara.release()
cv2.destroyAllWindows()
