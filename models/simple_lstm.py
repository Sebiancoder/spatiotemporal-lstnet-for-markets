import tensorflow as tf

class SimpleLSTM(tf.keras.Model):

    def __init__(self, hparams):

        super().__init__(name="simpleLSTM")

        self.hparams = hparams

        self.outputDense = tf.keras.layers.Dense(1, activation='relu')  

    def build(self, input_shape):

        self.hidden_lstms = []

        for i in range(self.hparams.Int("num_lstsms", default=2, min_value=0, max_value=5, step=1)):
        
            self.hidden_lstms.append(tf.keras.layers.LSTM(
                units=self.hparams.Int("lstm_units_{I}".format(I=i), default=128, min_value=16, max_value=256, step=16), 
                activation="tanh", 
                recurrent_activation="sigmoid", 
                recurrent_dropout=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05), 
                return_sequences=True, 
                stateful=False))

        self.outputLSTM = tf.keras.layers.LSTM(
                units=self.hparams.Int("output_lstm", default=128, min_value=16, max_value=256, step=16), 
                activation="tanh", 
                recurrent_activation="sigmoid", 
                recurrent_dropout=self.hparams.Float("dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05), 
                return_sequences=False, 
                stateful=False)

        super().build(input_shape)

    def call(self, inputs):
        
        x = inputs
        
        for lstm in self.hidden_lstms:

            x = lstm(x)

        return self.outputDense(self.outputLSTM(x))