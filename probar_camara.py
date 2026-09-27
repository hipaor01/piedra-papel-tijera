import cv2

camara = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not camara.isOpened():
    raise RuntimeError("No se ha podido abrir la cámara")

print("Pulsa la tecla q en la ventana de la cámara para cerrar.")

while True:
    correcto, imagen = camara.read()

    if not correcto:
        print("No se ha podido obtener la imagen")
        break

    cv2.imshow("Prueba de cámara", imagen)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()