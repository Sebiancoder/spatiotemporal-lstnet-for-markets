import tensorflow as tf
from models.simple_lstm import SimpleLSTM

#lstm with spatiotemporal relations encoded with lstm, prior to input.

class AttentionLSTM(tf.keras.Model):

    def __init__(self, hparams):

        super().__init__(name="Attention_LSTM")

        self.hparams = hparams

        self.temp_self_att = tf.keras.layers.Attention()

        self.spat_self_att = tf.keras.layers.Attention()

        self.lstm_component = SimpleLSTM(self.hparams)

    def build(self, input_shape):

        self.lstm_component.build(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        
        # data in its default form (num_sequences * timesteps * variables[aka the embedding for TSA - Temporal Self Attention])
        temporal_embedding = inputs

        #needs to be transposed for Spatial Self Attention to be shape (num_sequences * variables * timesteps[here the timesteps are the embedding for every variable])
        spatial_embedding = tf.transpose(inputs, perm=[0,2,1])

        tsa = self.temp_self_att([temporal_embedding, temporal_embedding])
        ssa = self.spat_self_att([spatial_embedding, spatial_embedding])

        org_shape_ssa = tf.transpose(ssa, perm=[0,2,1])

        return self.lstm_component(tf.add(tsa, org_shape_ssa))




