import tensorflow as tf

class LSTNet(tf.keras.Model):

    def __init__(self, num_heads = 5, key_dim = 5):

        super().__init__()

        temp_self_att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        spat_self_att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)    

    def build():
        pass

    def call():
        pass
