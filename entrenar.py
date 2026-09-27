from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODELO_BASE = "google/mobilenet_v2_1.0_224"
CARPETA_DATOS = Path("datos")
CARPETA_MODELO = Path("modelo_ppt")
EPOCAS = 8
TAMANO_LOTE = 16

# 1. Procesador asociado al modelo de Hugging Face
procesador = AutoImageProcessor.from_pretrained(MODELO_BASE)

media = procesador.image_mean
desviacion = procesador.image_std

# 2. Transformaciones de las imágenes
transformacion_entrenamiento = transforms.Compose(
    [
        transforms.CenterCrop((1000, 1000)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desviacion),
    ]
)

transformacion_evaluacion = transforms.Compose(
    [
        transforms.CenterCrop((1000, 1000)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desviacion),
    ]
)

# 3. Cargar las tres partes del conjunto de datos
datos_entrenamiento = datasets.ImageFolder(
    CARPETA_DATOS / "entrenamiento",
    transform=transformacion_entrenamiento,
)

datos_validacion = datasets.ImageFolder(
    CARPETA_DATOS / "validacion",
    transform=transformacion_evaluacion,
)

datos_prueba = datasets.ImageFolder(
    CARPETA_DATOS / "prueba",
    transform=transformacion_evaluacion,
)

categorias = datos_entrenamiento.classes
print("Categorías:", categorias)

id2label = {
    indice: nombre
    for indice, nombre in enumerate(categorias)
}

label2id = {
    nombre: indice
    for indice, nombre in id2label.items()
}

# 4. Preparar los cargadores de imágenes
cargador_entrenamiento = DataLoader(
    datos_entrenamiento,
    batch_size=TAMANO_LOTE,
    shuffle=True,
    num_workers=0,
)

cargador_validacion = DataLoader(
    datos_validacion,
    batch_size=TAMANO_LOTE,
    shuffle=False,
    num_workers=0,
)

cargador_prueba = DataLoader(
    datos_prueba,
    batch_size=TAMANO_LOTE,
    shuffle=False,
    num_workers=0,
)

# 5. Descargar el modelo y sustituir su clasificador
modelo = AutoModelForImageClassification.from_pretrained(
    MODELO_BASE,
    num_labels=len(categorias),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# 6. Congelar el modelo preentrenado
for parametro in modelo.parameters():
    parametro.requires_grad = False

# Entrenar solamente la nueva capa clasificadora
for parametro in modelo.classifier.parameters():
    parametro.requires_grad = True

# 7. Utilizar la GPU del Mac
dispositivo = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

modelo.to(dispositivo)
print("Dispositivo utilizado:", dispositivo)

# 8. Función de evaluación
def evaluar(modelo_actual, cargador):
    modelo_actual.eval()

    aciertos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            salida = modelo_actual(pixel_values=imagenes)
            predicciones = salida.logits.argmax(dim=1)

            aciertos += (predicciones == etiquetas).sum().item()
            total += etiquetas.size(0)

    return aciertos / total

# 9. Configuración del entrenamiento
criterio = nn.CrossEntropyLoss()

optimizador = torch.optim.AdamW(
    modelo.classifier.parameters(),
    lr=0.001,
)

mejor_precision = 0.0

# 10. Entrenamiento
for epoca in range(EPOCAS):
    modelo.eval()
    modelo.classifier.train()

    perdida_total = 0.0

    for imagenes, etiquetas in cargador_entrenamiento:
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        optimizador.zero_grad()

        salida = modelo(pixel_values=imagenes)
        perdida = criterio(salida.logits, etiquetas)

        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)

    perdida_media = perdida_total / len(datos_entrenamiento)
    precision_validacion = evaluar(
        modelo,
        cargador_validacion,
    )

    print(
        f"Época {epoca + 1}/{EPOCAS} | "
        f"Pérdida: {perdida_media:.4f} | "
        f"Validación: {precision_validacion:.2%}"
    )

    if precision_validacion > mejor_precision:
        mejor_precision = precision_validacion
        modelo.save_pretrained(CARPETA_MODELO)
        procesador.save_pretrained(CARPETA_MODELO)
        print("  Mejor modelo guardado")

# 11. Evaluación final con imágenes de prueba
mejor_modelo = AutoModelForImageClassification.from_pretrained(
    CARPETA_MODELO
)

mejor_modelo.to(dispositivo)

precision_prueba = evaluar(
    mejor_modelo,
    cargador_prueba,
)

print()
print("Entrenamiento terminado")
print(f"Mejor validación: {mejor_precision:.2%}")
print(f"Precisión de prueba: {precision_prueba:.2%}")
print(f"Modelo guardado en: {CARPETA_MODELO}")