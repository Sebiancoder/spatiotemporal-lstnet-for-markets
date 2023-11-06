import tensorflow as tf

class MLP(tf.keras.Model):
    #implementation of MLP (Multi Layer Perceptron aka your basic neural net)

    def __init__(self, hparams):

        super().__init__(name="simpleMLP")

        self.hparams = hparams

        self.dense_layers = []
        self.do_layers = []

        for i in range(self.hparams.Int("num_layers", default=3, min_value=2, max_value=5, step=1)):

            self.dense_layers.append(tf.keras.layers.Dense(self.hparams.Int("d_{I}".format(I=i), default=64, min_value=16, max_value=512, step=16), activation="relu"))
            self.do_layers.append(tf.keras.layers.Dropout(self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05)))

        self.outputDense = tf.keras.layers.Dense(1, activation="relu")

        self.flattener = tf.keras.layers.Flatten()

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, inputs):
        
        x = self.flattener(inputs)
        
        for layer_index in range(len(self.dense_layers)):

            x = self.dense_layers[layer_index](x)
            x = self.do_layers[layer_index](x)

        return self.outputDense(x)