import tensorflow as tf
from Model import Model
from Data_Loader import Data_Loader


# Instantiate the model
model = Model()

def train_data(folder: str):
    # ======================# Prepare Data #======================
    Data_LoaderObj = Data_Loader(augmentation=False)
    directory = f"data/{folder}"
    if not Data_LoaderObj.augmentation:
        imgs, labels = Data_LoaderObj.load_images_from_directory(directory=directory)

        # preproccess data
        return Data_LoaderObj.prepare_data(
            img=imgs, all_label=labels
        )

    else:
        # TODO: Implement data proccesing with augmented data
        pass

x_train, y_train_label, y_train_mask = train_data(folder="Train") # Datos de entrenamiento
x_val, y_val_label, y_val_mask = train_data(folder="Test") # Datos de validación

# ======================# End #======================

# ======================# Train model #======================
# Entrenar el modelo con early stopping
model.fit(
    x_train=x_train,
    y_train_label=y_train_label,
    y_train_mask=y_train_mask,
    x_val=x_val,
    y_val_label=y_val_label,
    y_val_mask=y_val_mask,
    epochs=100,  # Puedes ajustar este valor según sea necesario
    batch_size=36,  # Ajusta según los recursos disponibles
    patience=14  # Early stopping si no hay mejora en 5 epochs consecutivas
)