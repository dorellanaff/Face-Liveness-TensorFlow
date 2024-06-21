import cv2 as cv
import tensorflow as tf
from Model import Model
from deepface import DeepFace
import numpy as np

# Load the model
modelObj = Model()
model = modelObj.model

# Load weights for the model
model.load_weights("Weights/best_weights.h5")

# Read the image from file
img_path = r"C:\Users\dforellana\Desarrollo\Test\face\FaceAntiSpoofing\benchmarks\test\real\3.jpg"
img = cv.imread(img_path)
try:
    # Extraer caras de la imagen
    extracted_face = DeepFace.extract_faces(
        img,
        target_size=(224, 224),
        enforce_detection=True,
        detector_backend='mediapipe'
    )[0]
    
    x, y, w, h, _, _ = extracted_face["facial_area"].values()
    face = extracted_face["face"]
    face = np.expand_dims(face, axis=0)
    try:
        # Predicción del modelo
        mask, binary = model(face)
    except tf.errors.ResourceExhaustedError:
        print("Reduciendo el tamaño del lote debido a limitaciones de memoria")

    threshold = np.mean(mask)

    # Determinar y mostrar el resultado
    if threshold > 0.5:
        result = "Real"
    else:
        result = "Fake"

    # Mostrar el resultado en la consola
    print("Threshold:", threshold)
    print("Result:", result)

except Exception as e:
    print(e)