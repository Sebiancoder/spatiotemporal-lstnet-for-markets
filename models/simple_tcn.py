import tensorflow as tf
from tcn import TCN

class SimpleTCN(tf.keras.Model):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        self.outputDense = tf.keras.layers.Dense(1, activation='relu')  

    def build(self, input_shape):
        
        #calc num dilations for TCN, derived from equation seq_len ~= receptive field size  = 2^layers * (kernel_size - 1)
        num_dilations = np.floor(np.log2(input_shape[1]))

        self.tcn1 = TCN(
            nb_filters=self.hparams.Int("tcn_units1", default=128, min_value=16, max_value=256, step=16),
            kernel_size=2,
            dilations=np.geomspace(1, 2 ** num_dilations, num=num_dilations, endpoint=False),
            dropout_rate=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.5, step=0.05)
        )

        self.tcn2 = TCN(
            nb_filters=self.hparams.Int("tcn_units2", default=128, min_value=16, max_value=256, step=16),
            kernel_size=2,
            dilations=np.geomspace(1, 2 ** num_dilations, num=num_dilations, endpoint=False),
            dropout_rate=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.5, step=0.05)
        )

        super().build(input_shape)

    def call(self, inputs):
        
        x = self.lstm_layer(self.tcn2(self.tcn1(inputs)))

        return self.outputDense(x)