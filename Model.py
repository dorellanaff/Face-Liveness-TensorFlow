import tensorflow as tf
from Loss import PixWiseBCELoss
import numpy as np

class Model:
    def __init__(self, base_model=None) -> None:
        if base_model is None:
            base_model = tf.keras.applications.DenseNet121(include_top=False, input_shape=(224,224,3))
        self.features = base_model.layers[:171]
        x = self.features[-1].output
        out_feature_map = tf.keras.layers.Conv2D(1, (1, 1), padding='same', strides=1, activation='sigmoid')(x)
        out_map_flat = tf.keras.layers.Flatten()(out_feature_map)
        out_binary = tf.keras.layers.Dense(1, activation='sigmoid')(out_map_flat)
        input = self.features[0].input
        self.model = tf.keras.Model(inputs=input, outputs=[out_feature_map, out_binary])
        self.loss = PixWiseBCELoss(beta=0.5)
        self.optimizer = tf.keras.optimizers.Adam()

    def fit(
        self,
        x_train,
        y_train_label,
        y_train_mask,
        x_val,
        y_val_label,
        y_val_mask,
        epochs=100,
        batch_size=32,
        save_best_weights=True,
        shuffle=True,
        patience=5
    ):
        if shuffle:
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train_label = y_train_label[indices]
            y_train_mask = y_train_mask[indices]

        @tf.function
        def train_step(x_batch, y_mask_batch, y_label_batch):
            with tf.GradientTape() as tape:
                out_mask, out_label = self.model(x_batch, training=True)
                batch_loss = self.loss([y_mask_batch, y_label_batch], [out_mask, out_label])

            gradients = tape.gradient(batch_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return batch_loss

        @tf.function
        def val_step(x_batch, y_mask_batch, y_label_batch):
            out_mask, out_label = self.model(x_batch, training=False)
            batch_loss = self.loss([y_mask_batch, y_label_batch], [out_mask, out_label])
            return batch_loss

        epochs_losses = []
        val_losses = []
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            val_loss = 0

            # Training
            for batch in range(0, len(x_train), batch_size):
                x_batch = x_train[batch:batch + batch_size]
                y_mask_batch = y_train_mask[batch:batch + batch_size]
                y_label_batch = y_train_label[batch:batch + batch_size]
                batch_loss = train_step(x_batch, y_mask_batch, y_label_batch)
                epoch_loss += batch_loss.numpy()

            # Validation
            for batch in range(0, len(x_val), batch_size):
                x_batch = x_val[batch:batch + batch_size]
                y_mask_batch = y_val_mask[batch:batch + batch_size]
                y_label_batch = y_val_label[batch:batch + batch_size]
                batch_loss = val_step(x_batch, y_mask_batch, y_label_batch)
                val_loss += batch_loss.numpy()

            epoch_loss /= len(x_train) // batch_size
            val_loss /= len(x_val) // batch_size
            epochs_losses.append(epoch_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1} Loss: {epoch_loss}, Val Loss: {val_loss}")

            if save_best_weights:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    model_name = f"Weights/best_weights_xtrim_{epoch+1}.h5"
                    self.model.save_weights(model_name)
                    print(f"Weights of Model {model_name} saved with Val Loss: {val_loss}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
