import tensorflow as tf

class LastValueMLP(tf.keras.Model):
    #implementation of Last Value model (prediction is based on dense layer on the last timestep variables)

    def __init__(self, hparams):

        super().__init__(name="simpleMLP")

        self.dense = tf.keras.layers.Dense(1, activation="relu")

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs):
        
        x = inputs[:, -1, :]
        
        return self.dense(x)