import tensorflow as tf
from tcn import TCN
import numpy as np

#implementation of Self-Attention based LSTNet as described in this paper
#https://www.hindawi.com/journals/ijis/2023/9523230/

class LSTNet(tf.keras.Model):

    def __init__(self, hparams):

        super().__init__(name="LSTNet")

        self.hparams = hparams

        self.temp_self_att = tf.keras.layers.Attention()

        self.spat_self_att = tf.keras.layers.Attention()

        #final dense layer
        self.final_dense = tf.keras.layers.Dense(1, activation="relu")

        #flattener
        self.flattener = tf.keras.layers.Flatten()   

    def build(self, input_shape):

        #calc num dilations for TCN, derived from equation seq_len ~= receptive field size  = 2^layers * (kernel_size - 1)
        num_dilations = np.floor(np.log2(input_shape[1])).astype(int)

        dilations = np.geomspace(1, 2 ** num_dilations, num=num_dilations, endpoint=False).astype(int)

        #this is the MLP captioned "autoregression" in the diagram
        self.spat_mlp1 = tf.keras.layers.Dense(self.hparams.Int("spat_mlp1", default=64, min_value=16, max_value=256, step=16), activation="relu")
        self.spat_mlp2 = tf.keras.layers.Dense(self.hparams.Int("spat_mlp2", default=64, min_value=16, max_value=256, step=16), activation="relu")
        
        #TCN for "convolutional component" - this is all the paper mentions about this so I am not sure if they mean 1DConv or TCN, assuming TCN since it can't be worse?
        self.temporal_TCN = TCN(
            nb_filters=self.hparams.Int("tcn_units", default=64, min_value=16, max_value=256, step=16),
            kernel_size=2,
            dilations=dilations.reshape(len(dilations), 1),
            dropout_rate=self.hparams.Float("tcn_dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05),
            return_sequences=True
        )

        self.lstm_layer = tf.keras.layers.LSTM(
            units=input_shape[2], 
            activation="tanh", 
            recurrent_activation="sigmoid", 
            recurrent_dropout=self.hparams.Float("lstm_dropout", default=0.1, min_value=0.05, max_value=0.3, step=0.05), 
            return_sequences=False, 
            stateful=False) 

        super().build(input_shape)

    def call(self, inputs):

        #interestingly paper does not apply any other embedding to input data besides Identity Matrix (something we may want to play around with later)
        
        # data in its default form (num_sequences * timesteps * variables[aka the embedding for TSA - Temporal Self Attention])
        temporal_embedding = inputs

        #needs to be transposed for Spatial Self Attention to be shape (num_sequences * variables * timesteps[here the timesteps are the embedding for every variable])
        spatial_embedding = tf.transpose(inputs, perm=[0,2,1])

        tsa = self.temp_self_att([temporal_embedding, temporal_embedding])
        ssa = self.spat_self_att([spatial_embedding, spatial_embedding])

        convoluted_tsa = self.temporal_TCN(tsa)
        tsa_lstm = self.lstm_layer(convoluted_tsa)

        # "autoregression" not sure if i am doing this right
        postMLP_ssa = self.spat_mlp2(self.spat_mlp1(self.flattener(ssa)))

        return self.final_dense(tf.concat([postMLP_ssa, tsa_lstm], axis=1))




