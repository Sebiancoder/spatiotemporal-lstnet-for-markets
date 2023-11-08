import tensorflow as tf

class SimpleLSTM(tf.keras.Model):

    def __init__(self, hparams):

        super().__init__(name="simpleLSTM")

        self.hparams = hparams

        self.outputDense = tf.keras.layers.Dense(1, activation='relu')  

    def build(self, input_shape):
        
        self.lstm_layer = tf.keras.layers.LSTM(
            units=self.hparams.Int("lstm_units1", default=128, min_value=16, max_value=256, step=16), 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05), 
            return_sequences=False, 
            stateful=False)

        self.hidden_lstm = tf.keras.layers.LSTM(
            units=self.hparams.Int("lstm_units2", default=128, min_value=16, max_value=256, step=16), 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05), 
            return_sequences=True, 
            stateful=False)

        super().build(input_shape)

    def call(self, inputs):
        
        x = self.lstm_layer(self.hidden_lstm(inputs))

        return self.outputDense(x)