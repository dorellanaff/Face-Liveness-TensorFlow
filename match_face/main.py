import sys
from typing import Annotated
from fastapi import FastAPI, File, Request, UploadFile, Form, HTTPException, Header
import os
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deepface import DeepFace
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import cv2
from uuid import uuid4
import time
import traceback

# Configuración de CSRF
class CsrfSettings(BaseModel):
    secret_key: str = "123456789"
    csrf_max_age: int = 300  # Duración del token CSRF en segundos (5 minutos)

# Configuración global de CSRF
csrf_config = CsrfSettings()

origins = ["*"]

app = FastAPI()

#app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchResponse(BaseModel):
    message: str
    match: bool
    threshold: float

fake_db = {
    '0930499074': "daniel.png",
    '9999999999': "desconocido.png",
    '9999999991': "matthew.jpg",
    "1804469441": "1804469441.jpg",
    "0926789165": "0926789165.jpg",
    "0930917356": "0930917356.jpg",
    "0102413234": "0102413234.jpg",
}


fake_db_id_faces = {'6f5669df-90c5-4f21-b2a1-796fb9a51c8c': {'id_number': '0930499074', 'validated': False, 'result': None, 'csrf': 'ef4dd498-2a43-4b90-b4a6-afa737decc14', 'expiration_time': time.time() + 300}}

# Función para generar un nuevo token CSRF
def generate_csrf_token(id_face: str):
    token = str(uuid4())
    expiration_time = time.time() + csrf_config.csrf_max_age
    val = fake_db_id_faces.get(id_face, None)
    if val is None:
        raise HTTPException(status_code=404, detail="ID Verificacion no encontrado")
    fake_db_id_faces[id_face]['expiration_time'] = expiration_time
    fake_db_id_faces[id_face]['csrf'] = token
    return token

# Middleware para manejar la validación de CSRF
@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        token = request.headers.get("X-CSRF-Token")
        id_face = request.headers.get("id-face")
        if token is None and id_face is None:
            return JSONResponse(content={"message": "Not CSRF token or ID Face provided"}, status_code=403)
        
        if id_face not in fake_db_id_faces:
            return JSONResponse(content={"message": "ID Verificacion no encontrado"}, status_code=404)
        
        print(token, id_face)
        print(fake_db_id_faces)
        if token == fake_db_id_faces[id_face]['csrf']:
            if fake_db_id_faces[id_face]["expiration_time"] < time.time():
                return JSONResponse(content={"message": "Invalid CSRF time token"}, status_code=403)
        else:
            return JSONResponse(content={"message": "Invalid CSRF token"}, status_code=403)
    
    response = await call_next(request)
    return response

@app.get("/generate/{id_number}")
async def generate_id_face(id_number: str):
    if id_number in fake_db:
        # generate uuid
        
        myuuid = uuid4()
        id_face_dict = {
            "id_number": id_number,
            "validated": False,
            "result": None,
            "csrf": None,
            "expiration_time": None
        }
        myuuid = str(myuuid)
        fake_db_id_faces[myuuid] = id_face_dict
        
        return JSONResponse(content={"message": "ID Verificacion generado", "id_face": myuuid}, status_code=200)

    message = "No existe el registro solicitado"
    return JSONResponse(content={"message": message, "id_face": None}, status_code=400)

# Endpoint para obtener un nuevo token CSRF
@app.get("/csrf-token/{id_face}")
def get_csrf_token(id_face: str):
    token = generate_csrf_token(id_face=id_face)
    return JSONResponse(content={"csrf_token": token})

@app.post("/match/", response_model=MatchResponse)
async def predict_face(request: Request, file: UploadFile = File(...)):
    id_number = None
    try:
        id_face = request.headers.get("id-face")
        
        check = fake_db_id_faces.get(id_face)
        
        id_number = check["id_number"]
        
        path_temp = f"{id_face}_{file.filename}"
        path_file_temp = os.path.join("temp", path_temp)
        
        # Guardar la imagen temporalmente para su procesamiento
        with open(path_file_temp, "wb") as img_file:
            img_file.write(file.file.read())
        
        path_validate_file = fake_db[id_number]
        path_validate_file = os.path.join("db", path_validate_file)
        
        # Verificar similitud usando FaceNet con distancia euclidiana
        """result_euclidean = DeepFace.verify(
            img1_path=path_validate_file, img2_path=path_file_temp, model_name='ArcFace', 
            detector_backend="mtcnn", distance_metric='euclidean', anti_spoofing=True)"""
        
        face_objs = DeepFace.extract_faces(img_path=path_file_temp, anti_spoofing=True)
        assert len(face_objs) == 1, f"Expected 1 face, but found {len(face_objs)} faces."
        assert face_objs[0].get('is_real') is True, f"Face is not real."
                
        # Verificar similitud usando FaceNet con distancia euclidiana
        result_euclidean = DeepFace.verify(
            img1_path=path_validate_file, img2_path=path_file_temp, anti_spoofing=True)
        '''
        1    ArcFace            mtcnn       euclidean       70.110112         235           47      0.166667
        0    ArcFace            mtcnn          cosine       58.241266         278            4      0.014184
        5    Facenet            mtcnn       euclidean       54.890929         186           96      0.340426  
        3    ArcFace              ssd       euclidean       53.663358         207           75      0.265957  
        '''
        print(result_euclidean)
        
        threshold = 0
        
        threshold = threshold + 5 if result_euclidean["verified"] else 0
        
        f_match = True if threshold >= 5 else False
        message = "Verificacion exitosa" if f_match else "Verificacion fallida"
        
        threshold = result_euclidean["distance"] / result_euclidean["threshold"]
        response = MatchResponse(message=message, match=f_match, threshold=threshold)
        
        '''fake_db_id_faces[id_face] = {
            "id_number": id_number,
            "validated": True,
            "result": response.model_dump()
        }'''
        fake_db_id_faces[id_face]['id_number'] = id_number
        fake_db_id_faces[id_face]['validated'] = True
        fake_db_id_faces[id_face]['result'] = response.model_dump()
        
        # Token válido, eliminarlo después de su uso
        fake_db_id_faces[id_face]['csrf'] = None
        return response

    except Exception as e:
        etype, evalue, tb = sys.exc_info()
        print("Error capturado:", e)
        
        # Recorre la cadena de excepciones para encontrar el primer error
        original_exception = e
        while original_exception.__cause__:
            original_exception = original_exception.__cause__
            
        # Imprime el primer error
        etype, evalue, tb = type(original_exception), original_exception, original_exception.__traceback__
        formatted_exception = traceback.format_exception_only(etype, evalue)
        first_error_message = formatted_exception[0].strip()  # Obtener el primer mensaje de error
        print("Primer mensaje de error:", first_error_message)
        
        response = MatchResponse(message=str(e), match=False, threshold=0)
        
        '''fake_db_id_faces[id_face] = {
            "id_number": id_number,
            "validated": True,
            "result": response.model_dump()
        }'''
        fake_db_id_faces[id_face]['id_number'] = id_number
        fake_db_id_faces[id_face]['validated'] = True
        fake_db_id_faces[id_face]['result'] = response.model_dump()
        return response

def save_image(file: UploadFile):
    uuid_str = str(uuid4())
    
    path_temp = f"{uuid_str}_{file.filename}"
    path_file_temp = os.path.join("temp", path_temp)
    
    # Guardar la imagen temporalmente para su procesamiento
    with open(path_file_temp, "wb") as img_file:
        img_file.write(file.file.read())
    
    return path_file_temp

@app.post("/match_files/")
async def predict_face(files: list[UploadFile]):

    #validate only two files
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Se esperan dos archivos")
    
    file_1 = files[0]
    file_2 = files[1]

    model_threshold = 0
    try:
        
        file_1_path = save_image(file=file_1)
        file_2_path = save_image(file=file_2)
        
        # Verificar similitud usando FaceNet con distancia euclidiana
        result_vgg_face = DeepFace.verify(
            file_1_path, 
            file_2_path, 
            model_name='VGG-Face', 
            distance_metric='euclidean',
            detector_backend='retinaface'
        )
        model_threshold = model_threshold + (5 if result_vgg_face["verified"] else 0)
        print('result_vgg_face', result_vgg_face, model_threshold)
        
        # Verificar los otros modelos
        result_arcfac = DeepFace.verify(
            file_1_path, 
            file_2_path, 
            model_name='ArcFace', 
            distance_metric='euclidean',
            detector_backend='mediapipe'
        )
        model_threshold = model_threshold + (2.49 if result_arcfac["verified"] else 0)
        print('result_arcfac', result_arcfac, model_threshold)
        
        result_facenet = DeepFace.verify(
            file_1_path, 
            file_2_path, 
            model_name='Facenet',
            distance_metric='cosine',
            detector_backend='retinaface'
        )
        model_threshold = model_threshold + (2.49 if result_facenet["verified"] else 0)
        print('result_facenet', result_facenet, model_threshold)
        
        verify = True if model_threshold >= 5 else False
        message = "Verificacion exitosa" if verify else "Verificacion fallida"
        response = {
            'message': message,
            'verify': verify,
            'model_threshold': model_threshold
        }
        
        return response

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))