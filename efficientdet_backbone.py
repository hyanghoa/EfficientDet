import tensorflow as tf

class EfficientNet(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(EfficientNet, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

layer = EfficientNet(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)