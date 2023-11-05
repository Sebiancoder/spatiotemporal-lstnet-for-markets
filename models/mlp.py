import tensorflow as tf

class MLP(tf.keras.Model):
    #implementation of MLP (Multi Layer Perceptron aka your basic neural net)

    def __init__(self, hparams):

        super().__init__(name="simpleMLP")

        self.hparams = hparams

        self.dense_layers = []

        for i in range(self.hparams.int("d1", default=3, min_value=2, max_value=5, step=1)):

            self.dense_layers.append(tf.keras.layers.Dense(self.hparams.int("d_{I}".format(I=i), default=64, min_value=16, max_value=512, step=16), activation="relu"))

        self.outputDense = tf.keras.layers.Dense(1, activation="relu")

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs):
        
        x = inputs
        
        for layer in self.dense_layers:

            x = layer(x)

        return self.outputDense(x)