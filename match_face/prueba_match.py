import os
from deepface import DeepFace
import pandas
import openpyxl

import os


path_dir_images = r'C:\Users\dforellana\Desarrollo\Test\face\repo_avanti'
models = ["DeepID", 'Facenet', 'VGG-Face', 'ArcFace', "OpenFace", "DeepFace"]
detector_backends = ["opencv", "ssd", "mtcnn", "dlib", "retinaface", "mediapipe"]
distance_metrics = ["cosine", "euclidean"]

list_models = {model: [] for model in models}

list_all = []

# get folders of path
path_dir = os.path.join(path_dir_images, "images")
path_dir = os.path.join(path_dir, 'test')

ok_path_dir = os.path.join(path_dir, "ok")
fail_path_dir = os.path.join(path_dir, "fail")

ok_list = os.listdir(ok_path_dir)
fail_list = os.listdir(fail_path_dir)

stop = r'C:\Users\dforellana\Desarrollo\Test\face\repo_avanti\images\test\ok\1314977040_149589'

def main(list_folders: list, dir_path: str):
    for x, list_folder in enumerate(list_folders):
        
        list_folder_path_tmp = os.path.join(dir_path, list_folder)
        
        if list_folder_path_tmp == stop:
            print(x, len(list_folders))
            break
        

if __name__ == "__main__":
    main(list_folders=ok_list, dir_path=ok_path_dir)
    
    """
    # save list_models in excel
    for model in models:
        df_models = pandas.DataFrame(list_models[model])
        df_models.to_excel(f'{model}.xlsx', index=False, header=True)
    """
    
    
    df_models = pandas.DataFrame(list_all)
    df_models.to_excel(f'All.xlsx', index=False, header=True)
    