from deepface import DeepFace
import sys

# Carga las imágenes que deseas comparar
img1 = 'data/fotos/image1.png'
img2 = 'data/fotos/1.jpg'

# Umbral de distancia máxima
threshold = 10

'''
# Encontrar la representación del rostro en una imagen
representation = DeepFace.represent(img1, model_name="Facenet")
print("Face representation: ", representation)
representation = DeepFace.represent(img2, model_name="Facenet")
print("Face representation: ", representation)
'''

# Verificar similitud usando FaceNet con distancia euclidiana
result_euclidean = DeepFace.verify(img1, img2, model_name='Facenet', distance_metric='euclidean')
print("Euclidean Distance Result:", result_euclidean)
distance = result_euclidean["distance"]
# Calcular el porcentaje de similitud
similarity_percentage = (distance / threshold) * 100
# Asegurarse de que el porcentaje esté dentro de los límites de 0 a 100
similarity_percentage = max(0, min(100, similarity_percentage))

print(f"Similitud: {similarity_percentage:.2f}%")
'''
# Verificar similitud usando FaceNet con distancia coseno
result_cosine = DeepFace.verify(img1, img2, model_name='Facenet', distance_metric='cosine')
print("Cosine Distance Result:", result_cosine)
'''


# Utiliza VGG-Face para comparar las imágenes
model_name='VGG-Face'
result = DeepFace.verify(img1, img2, model_name=model_name, distance_metric='cosine')
print(result)

model_name='Facenet'
result = DeepFace.verify(img1, img2, model_name=model_name, distance_metric='euclidean')
# Imprime el resultado
print(result)

model_name='ArcFace'
result = DeepFace.verify(img1, img2, model_name=model_name)
# Imprime el resultado
print(result)
sys.exit(-1)

model_name='OpenFace'
result = DeepFace.verify(img1, img2, model_name=model_name)
# Imprime el resultado
print(result)

model_name='DeepFace'
#result = DeepFace.verify(img1, img2, model_name=model_name)
#print(result)

model_name='DeepID'
result = DeepFace.verify(img1, img2, model_name=model_name)
# Imprime el resultado
print(result)

model_name='Dlib'
result = DeepFace.verify(img1, img2, model_name=model_name)
# Imprime el resultado
print(result)
