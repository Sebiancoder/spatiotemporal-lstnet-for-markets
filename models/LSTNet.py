import tensorflow as tf
from tcn import TCN

#implementation of Self-Attention based LSTNet as described in this paper
#https://www.hindawi.com/journals/ijis/2023/9523230/

class LSTNet(tf.keras.Model):

    def __init__(self, units=128, dropout_rate=0.1, spat_mlp1_units=64):

        super().__init__()

        self.temp_self_att = tf.keras.layers.Attention()

        self.spat_self_att = tf.keras.layers.Attention()

        #this is the MLP captioned "autoregression" in the diagram
        self.spat_mlp1 = tf.keras.layers.Dense(spat_mlp1_units, activation="relu")
        self.spat_mlp2 = tf.keras.layers.Dense(spat_mlp2_units, activation="relu")

        #final dense layer
        self.final_dense = tf.keras.layers.Dense(1, activation="relu")   

    def build(self, input_shape):

        #calc num dilations for TCN, derived from equation seq_len ~= receptive field size  = 2^layers * (kernel_size - 1)
        num_dilations = np.floor(np.log2(input_shape[1]))
        
        #TCN for "convolutional component" - this is all the paper mentions about this so I am not sure if they mean 1DConv or TCN, assuming TCN since it can't be worse?
        self.temporal_TCN = TCN(
            nb_filters=units,
            kernel_size=2,
            dilations=np.geomspace(1, 2 ^ num_dilations, num=num_dilations, endpoint=False),
            dropout_rate=dropout_rate
        )

        self.lstm_layer = tf.keras.layers.LSTM(
            units=128, 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=0.1, 
            return_sequences=False, 
            stateful=False) 

        super().build(input_shape)

    def call(self, inputs):

        #interestingly paper does not apply any other embedding to input data besides Identity Matrix (something we may want to play around with later)
        
        # data in its default form (num_sequences * timesteps * variables[aka the embedding for TSA - Temporal Self Attention])
        temporal_embedding = inputs

        #needs to be transposed for Spatial Self Attention to be shape (num_sequences * variables * timesteps[here the timesteps are the embedding for every variable])
        spatial_embedding = tf.transpose(inputs, perm=[0,2,1])

        tsa = self.temp_self_att(temporal_embedding)
        ssa = self.spat_self_att(spatial_embedding)

        convoluted_tsa = self.temporal_TCN(tsa)
        tsa_lstm = self.lstm_layer(convoluted_tsa)

        postMLP_ssa = self.spat_mlp2(self.spat_mlp1(ssa))

        retransposed_spatial = tf.transpose(postMLP_ssa, perm=[0, 2, 1])

        return self.final_dense(tf.concat([retransposed_spatial, tsa_lstm], axis=1))




