import tensorflow as tf
from kerastuner import HyperModel

class SimpleLSTM(tf.keras.Model):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        self.outputDense = tf.keras.layers.Dense(1, activation='relu')  

    def build(self, input_shape):
        
        self.lstm_layer = tf.keras.layers.LSTM(
            units=self.hparams.Int("lstm_units1", default=64, min_value=16, max_value=256, step=16), 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.5, step=0.05), 
            return_sequences=False, 
            stateful=False)

        self.hidden_lstm = tf.keras.layers.LSTM(
            units=self.hparams.Int("lstm_units2", default=64, min_value=16, max_value=256, step=16), 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.5, step=0.05), 
            return_sequences=True, 
            stateful=False)

        super().build(input_shape)

    def call(self, inputs):
        
        x = self.lstm_layer(self.hidden_lstm(inputs))

        return self.outputDense(x)

    def get_search_space(self):
        #return the hyperparameter search space for the model

        return {
            "lstm1_units": [16, 32, 64, 128],
            "lstm2_units": [16, 32, 64, 128],
            "dropout": [0.05, 0.1, 0.2, 0.3]
        }