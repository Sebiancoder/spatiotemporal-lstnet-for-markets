import tensorflow as tf

class SimpleLSTM(tf.keras.Model):

    def __init__(self, num_heads = 5, key_dim = 5):

        super().__init__()

        self.denseLayer = tf.keras.layers.Dense(16, activation='sigmoid')  

    def build(self, inputs):
        
        self.lstm_layer = tf.keras.layers.LSTM(
            units=64, 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=0.2, 
            return_sequences=False, 
            stateful=False)

    def call(self, inputs):
        
        x = self.lstm_layer(inputs)

        x = self.denseLayer(x)

        return x