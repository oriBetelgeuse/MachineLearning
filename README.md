# MLmodels
В данном разделе реализованы некоторые архитектуры нейронных сетей.

## TimeSeriesTransformer(num_layers, d_model, num_heads, d_feed_forward, target, rate=0.1)
Находится в ветке transformer. По факту представляет собой encoder классического трансформера с отсутствующими слоями Embedding и Positional Encoding. Вместо decoder используется обычный Dense слой с тождественной функцией активации. Реализацию классического трансформера с описанием каждой части можно посмотреть здесь https://www.tensorflow.org/tutorials/text/transformer. Данная реализация предназначена для обработки многомерных временных рядов.

### Input Parameters

> num_layers : int
>> Количество слоёв TransformerEncoderLayer.

> d_model : int
>> Количество нейронов в линейных слоях MultiHeadAttention.

> num_heads : int
>> Количество частей ("голов") в MultiHeadAttention.

> d_feed_forward : int
>> Количество нейронов в Dense слоях TransformerEncoderLayer.

> rate : [0.0, 1.0]
>> Rate в Dropout слоях.

> target : int
>> Размерность выхода сети.
