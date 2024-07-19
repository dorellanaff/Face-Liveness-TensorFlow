from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import numpy as np
#import tensorflow as tf
from Model import Model
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import os
cwd = os.getcwd()

origins = ["*"]

app = FastAPI()

# app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#0.44837379455566406
#0.6874078512191772
# Cargar el modelo al iniciar la aplicación
modelObj = Model()
model = modelObj.model
path_model = os.path.join(cwd, 'Weights')
path_model = os.path.join(path_model, 'best_weights_xtrim_1.h5')
model.load_weights(path_model) #"Weights/best_weights_xtrim_better.h5"

# best_weights_xtrim_better - 0.050765
# best_weights_xtrim_epoch1 - 0.050765
# best_weights_xtrim_1      - 0.089974
# best_weights              - 0.118865
# best_weights_checkpoint   - 0.056415

class PredictionResponse(BaseModel):
    message: str
    threshold: float

@app.post("/predict/", response_model=PredictionResponse)
async def predict_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        # Convertir la imagen a un formato de OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extraer caras de la imagen usando DeepFace
        extracted_faces = DeepFace.extract_faces(
            img,
            #target_size=(224, 224),
            #enforce_detection=True,
            #detector_backend='mediapipe',
            anti_spoofing= True
        )
        
        print(extracted_faces)

        if not extracted_faces:
            return JSONResponse(content={"message": "No se detectó ninguna cara en la imagen"}, status_code=400)
        
        extracted_face = extracted_faces[0]
        x, y, w, h, _, _  = extracted_face["facial_area"].values()
        face = extracted_face["face"]
        face = np.expand_dims(face, axis=0)

        # Predicción del modelo
        mask, binary = model(face)
        threshold = np.mean(mask)

        message = "Real" if threshold > 0.4 else "Fake"
        response = PredictionResponse(message=message, threshold=threshold)
        return response

    except Exception as e:
        print(e)
        response = PredictionResponse(message=str(e), threshold=0)
        return response
