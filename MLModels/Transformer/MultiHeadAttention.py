import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.weighted_query = tf.keras.layers.Dense(d_model)
        self.weighted_key = tf.keras.layers.Dense(d_model)
        self.weighted_value = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, data, batch_size):
        data = tf.reshape(data, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(data, perm=[0, 2, 1, 3])

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask):
        attention_func = tf.matmul(query, key, transpose_b=True)
        scaling = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_func = attention_func / tf.math.sqrt(scaling)
        if mask is not None:
            scaled_attention_func += mask * -1e9
        attention_weights = tf.nn.softmax(scaled_attention_func, axis=-1)
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]

        query = self.weighted_query(query)
        key = self.weighted_key(key)
        value = self.weighted_value(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights
