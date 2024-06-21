import tensorflow as tf

class PixWiseBCELoss(tf.keras.losses.Loss):
    def __init__(self, beta=0.5, name='pixwise_bce_loss'):
        super(PixWiseBCELoss, self).__init__(name=name)
        self.beta = beta
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_true_mask, y_true_label = y_true
        y_pred_mask, y_pred_label = y_pred
        mask_loss = self.bce(y_true_mask, y_pred_mask)
        label_loss = self.bce(y_true_label, y_pred_label)
        return self.beta * mask_loss + (1 - self.beta) * label_loss
