import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from deepface import DeepFace
import pandas as pd

path_dir_images = r'C:\Users\dforellana\Desarrollo\Test\face\repo_avanti'
models = ['Facenet', 'ArcFace', "VGG-Face"] # "OpenFace" -- muchos fallas
detector_backends = ["ssd", "mtcnn", "retinaface", "mediapipe", "opencv"]
distance_metrics = ["cosine", "euclidean"]

list_models = {model: [] for model in models}

list_all = []

# get folders of path
path_dir = os.path.join(path_dir_images, "images")
#path_dir = os.path.join(path_dir, 'test')

ok_path_dir = os.path.join(path_dir, "ok")
fail_path_dir = os.path.join(path_dir, "fail")

ok_list = os.listdir(ok_path_dir)
fail_list = os.listdir(fail_path_dir)

def ensemble_verify(args):
    img1_path, img2_path, ci, _id, model, detector_backend, distance_metric = args
    try:
        result = DeepFace.verify(
            silent=True,
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model,
            detector_backend=detector_backend,
            distance_metric=distance_metric
        )
        verified = result["verified"]
        distance = result["distance"]
        threshold = result["threshold"]
        model_result = {
            "ci": ci,
            "id": _id,
            "verified": verified,
            "distance": distance,
            "threshold": threshold,
            "detector_backend": detector_backend,
            "distance_metric": distance_metric,
            "model": model
        }
    except:
        model_result = {
            "ci": ci,
            "id": _id,
            "verified": False,
            "distance": 0,
            "threshold": 0,
            "detector_backend": detector_backend,
            "distance_metric": distance_metric,
            "model": model
        }
    return model_result

def predict_face(args):
    path_folder, ci, _id, count = args
    results = []
    try:
        list_files = os.listdir(path_folder)
        if len(list_files) < 2:
            raise ValueError("Not enough files in folder to compare")
        file_1_path = os.path.join(path_folder, list_files[0])
        file_2_path = os.path.join(path_folder, list_files[1])
        for model in models:
            for detector_backend in detector_backends:
                for distance_metric in distance_metrics:
                    results.append(ensemble_verify((file_1_path, file_2_path, ci, _id, model, detector_backend, distance_metric)))
    except Exception as e:
        print(f'ci: {ci}, id: {_id}, error: {str(e)}')
    finally:
        print(f'{count}.')
    return results

def main(list_folders: list, dir_path: str):
    args_list = []
    for count, folder in enumerate(list_folders):
        split_folder = folder.split('_')
        if len(split_folder) >= 2:  # Asegurarse de que hay al menos 2 elementos
            ci, _id = split_folder[0], split_folder[1]
            path_folder = os.path.join(dir_path, folder)
            args_list.append((path_folder, ci, _id, count))
            
    # LIMIT TO 10
    # args_list = args_list[30:60]

    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        future_to_args = {executor.submit(predict_face, args): args for args in args_list}
        for future in as_completed(future_to_args):
            result = future.result()
            list_all.extend(result)

if __name__ == "__main__":
    main(list_folders=ok_list, dir_path=ok_path_dir)
    
    df_models = pd.DataFrame(list_all)
    df_models.to_excel(f'All_2.xlsx', index=False, header=True)
