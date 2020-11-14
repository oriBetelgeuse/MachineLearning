import tensorflow as tf

from MultiHeadAttention import MultiHeadAttention


class TimeSeriesTransformer(tf.keras.models.Model):

    def __init__(self, num_layers, d_model, num_heads, d_feed_forward, target, rate=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_feed_forward, rate)
        self.final_layer = tf.keras.layers.Dense(target)

    def call(self, data, training=None, mask=None):
        encoder_output = self.encoder(data, training, mask)
        final_output = self.final_layer(encoder_output)

        return final_output


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, d_feed_forward, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, d_feed_forward, rate)
                               for _ in range(num_layers)]

    def call(self, data, training=None, mask=None):
        for i in range(self.num_layers):
            data = self.encoder_layers[i](data, training, mask)

        return data


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, d_feed_forward, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = self.point_wise_feed_forward_network(d_model, d_feed_forward)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    @staticmethod
    def point_wise_feed_forward_network(d_model, d_feed_forward):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_feed_forward)
        ])

    def call(self, data, training=None, mask=None):
        attention_output, _ = self.multi_head_attention(data, data, data, mask)
        attention_output = self.dropout1(attention_output, traning=training)
        output1 = self.norm1(attention_output + data)

        feed_forward_output = self.feed_forward(output1)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        output2 = self.norm2(output1 + feed_forward_output)

        return output2
