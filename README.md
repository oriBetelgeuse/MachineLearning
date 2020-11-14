# MLmodels
В данном разделе реализованы некоторые архитектуры нейронных сетей.

## TimeSeriesTransformer
Находится в ветке transformer. По факту представляет собой encoder классического трансформера с отсутствующими слоями Embedding и Positional Encoding. Вместо decoder используется обычный Dense слой. Реализацию классического трансформера с описанием каждой части можно посмотреть здесь https://www.tensorflow.org/tutorials/text/transformer. Данная реализация предназначена для обработки многомерных временных рядов.

### Input Parameters
