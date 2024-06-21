import tensorflow as tf
import numpy as np
import os
from PIL import Image
import imgaug.augmenters as iaa

class Data_Loader:
    def __init__(
        self, target_size=(224, 224), augmentation=False, rescale=1.0 / 255
    ) -> None:
        self.target_size = target_size
        self.augmentation = augmentation
        self.rescale = rescale

    def load_images_from_directory(self, directory):
        image_data = []
        labels = []

        # Augmentation configuration
        if self.augmentation:
            seq = iaa.Sequential(
                [
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    iaa.Sometimes(
                        0.5, iaa.GaussianBlur(sigma=(0, 0.5))
                    ),  # Gaussian blur with random sigma
                    iaa.ContrastNormalization((0.75, 1.5)),  # contrast normalization
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),  # additive Gaussian noise
                    iaa.Multiply(
                        (0.8, 1.2), per_channel=0.2
                    ),  # multiply each pixel with random values
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scaling of images
                        translate_percent={
                            "x": (-0.2, 0.2),
                            "y": (-0.2, 0.2),
                        },  # translation of images
                        rotate=(-25, 25),  # rotation of images
                        shear=(-8, 8),  # shearing of images
                    ),
                ],
                random_order=True,
            )  # apply augmentations in random order

        # Iterate over each class directory
        for label, class_name in enumerate(os.listdir(directory)):
            class_dir = os.path.join(directory, class_name)

            # Iterate over each image in the class directory
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path)
                image = image.resize(self.target_size)
                image = np.array(image)

                if self.augmentation:
                    # Apply augmentation
                    image_augmented = seq.augment_image(image)
                    image_data.append(image_augmented)
                else:
                    image_data.append(image)

                # Apply rescaling
                image_data[-1] = image_data[-1] * self.rescale
                labels.append(label)

        return np.array(image_data), np.array(labels)

    # prepare loaded images for model
    def prepare_data(self, img, all_label, smoothing=True):
        images = []
        labels = []
        masks = []
        label_weight = 0.99 if smoothing else 1.0
        for image, label in zip(img, all_label):
            label_tensortype = tf.cast(label, dtype=tf.float32)
            label = label_tensortype
            map_size = 14  # this is size of our feature mao that will produce in model
            if label == 0:
                mask = np.ones((1, map_size, map_size), dtype=np.float32) * (
                    1 - label_weight
                )
            else:
                mask = np.ones((1, map_size, map_size), dtype=np.float32) * (
                    label_weight
                )

            images.append(image)
            labels.append(label)
            masks.append(mask)
        # change list to numpy array
        x_train = np.array(images)
        y_train_mask = np.array(masks)
        y_train_label = np.array(labels)
        # prepare dimension
        y_train_mask = np.squeeze(y_train_mask, axis=1)
        y_train_mask = np.expand_dims(y_train_mask, axis=-1)
        y_train_label = np.expand_dims(y_train_label, axis=-1)

        return x_train, y_train_label, y_train_mask
    
# Create Loss function
class PixWiseBCELoss(tf.keras.losses.Loss):
    def __init__(self, beta=0.5):
        super(PixWiseBCELoss, self).__init__()
        self.criterion = tf.keras.losses.BinaryCrossentropy()
        self.beta = beta

    def call(self, y_true, y_pred):
        target_mask, target_label = y_true
        net_mask, net_label = y_pred

        pixel_loss = self.criterion(target_mask, net_mask)
        binary_loss = self.criterion(target_label, net_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss

class Model:
    def __init__(self) -> None:
        # Load DenseNet121
        dense = tf.keras.applications.DenseNet121(include_top=False, input_shape=(224,224,3))
        self.features = dense.layers
        # Extract desirable layers from DenseNet121
        enc = self.features[:171]
        x = enc[-1].output  # Get output of the last extracted layer
        out_feature_map = tf.keras.layers.Conv2D(1,(1,1), padding='same', strides=1, activation='sigmoid')(x)
        out_map_flat = tf.keras.layers.Flatten()(out_feature_map)
        out_binary = tf.keras.layers.Dense(1, activation='sigmoid')(out_map_flat)
        # Define a new model using the extracted layers and optionally your own layers
        input = enc[0].input
        self.model = tf.keras.Model(inputs=input, outputs=[out_feature_map, out_binary])
        self.loss = PixWiseBCELoss(beta=0.5)
        self.optimizer = tf.keras.optimizers.Adam()

    # Define fit method to Train model
    def fit(
        self,
        x_train,
        y_train_label,
        y_train_mask,
        epochs=100,
        batch_size=32,
        save_best_weights=True,
        shuffle=True,
    ):
        if shuffle == True:
            # shuffle data
            # we shuflled data first
            indices = np.arange(x_train.shape[0])
            # we shuffled indices
            np.random.shuffle(indices)
            # reassign data(features dataset) and y dataset
            x_train = x_train[indices]
            y_train_label = y_train_label[indices]
            y_train_mask = y_train_mask[indices]

        # using GradientTape to update weights
        # Using tf.function decorator on train_step could potentially improve performance,
        # I would recommend experimenting with both options: with and without tf.function, and measure the performance difference.
        @tf.function
        def train_step(x_batch, y_mask_batch, y_label_batch):
            with tf.GradientTape() as tape:
                out_mask, out_label = self.model(x_batch)
                batch_loss = self.loss(
                    [y_mask_batch, y_label_batch], [out_mask, out_label]
                )

            gradients = tape.gradient(batch_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            return batch_loss

        # Training loop
        epochs = epochs
        epochs_losses = []
        batch_size = batch_size
        num_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size

                x_batch = x_train[start:end]
                y_mask_batch = y_train_mask[start:end]
                y_label_batch = y_train_label[start:end]

                batch_loss = train_step(x_batch, y_mask_batch, y_label_batch)
                epoch_loss += batch_loss.numpy()

            scaled_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} Loss: {scaled_loss}")
            if save_best_weights:
                epochs_losses.append(scaled_loss)
                # save best weights of model
                if scaled_loss <= min(epochs_losses):
                    print(f"Weights of Model saved with Loss: {scaled_loss}")
                    self.model.save_weights("Weights/best_weights.h5")
                    
# Instantiate the model
model = Model()

# ======================# Prepare Data #======================
Data_LoaderObj = Data_Loader(augmentation=False)
directory = "data/images/Train"
if not Data_LoaderObj.augmentation:
    imgs, labels = Data_LoaderObj.load_images_from_directory(directory=directory)
    # preproccess data
    x_train, y_train_label, y_train_mask = Data_LoaderObj.prepare_data(
        img=imgs, all_label=labels
    )

else:
    # TODO: Implement data proccesing with augmented data
    pass

# ======================# End #======================

# ======================# Train model #======================

model.fit(
    x_train=x_train,
    y_train_label=y_train_label,
    y_train_mask=y_train_mask,
    epochs=5,
    batch_size=2, # you can change this based on Computation power available for you
)