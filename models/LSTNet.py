import tensorflow as tf
from tcn import TCN

#implementation of Self-Attention based LSTNet as described in this paper
#https://www.hindawi.com/journals/ijis/2023/9523230/

class LSTNet(tf.keras.Model):

    def __init__(self, units=128, dropout_rate=0.1):

        super().__init__()

        self.temp_self_att = tf.keras.layers.Attention()

        self.spat_self_att = tf.keras.layers.Attention()   

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


