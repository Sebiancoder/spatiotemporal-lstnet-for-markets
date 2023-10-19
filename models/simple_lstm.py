import tensorflow as tf

class SimpleLSTM(tf.keras.Model):

    def __init__(self, num_heads = 5, key_dim = 5):

        super().__init__()

        self.outputDense = tf.keras.layers.Dense(1, activation='relu')  

    def build(self, input_shape):
        
        self.lstm_layer = tf.keras.layers.LSTM(
            units=128, 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=0.1, 
            return_sequences=False, 
            stateful=False)

        self.hidden_lstm = tf.keras.layers.LSTM(
            units=128, 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=0.1, 
            return_sequences=True, 
            stateful=False)

        super().build(input_shape)

    def call(self, inputs):
        
        x = self.lstm_layer(self.hidden_lstm(inputs))

        return self.outputDense(x)