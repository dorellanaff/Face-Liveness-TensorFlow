from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import numpy as np
import tensorflow as tf
from Model import Model

app = FastAPI()

# Cargar el modelo al iniciar la aplicación
modelObj = Model()
model = modelObj.model
model.load_weights("Weights/best_weights_checkpoint.h5")

class PredictionResponse(BaseModel):
    message: str
    threshold: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_face(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        # Convertir la imagen a un formato de OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extraer caras de la imagen usando DeepFace
        extracted_faces = DeepFace.extract_faces(
            img,
            target_size=(224, 224),
            enforce_detection=True,
            detector_backend='mediapipe'
        )

        if not extracted_faces:
            return JSONResponse(content={"message": "No se detectó ninguna cara en la imagen"}, status_code=400)
        
        extracted_face = extracted_faces[0]
        x, y, w, h, _, _  = extracted_face["facial_area"].values()
        face = extracted_face["face"]
        face = np.expand_dims(face, axis=0)

        # Predicción del modelo
        mask, binary = model(face)
        threshold = np.mean(mask)

        message = "Real" if threshold > 0.5 else "Fake"
        response = PredictionResponse(message=message, threshold=threshold)
        return response

    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
