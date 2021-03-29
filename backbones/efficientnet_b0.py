import tensorflow as tf


class EfficientNetB0(tf.keras.Model):

    def __init__(self):
        super(EfficientNetB0, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(1,),
                                initializer='random_normal',
                                trainable=True)

    def call(self, input_tensor, training=False):
        input_tensor = tf.matmul(input_tensor, self.w) + self.b
        x = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=3,
                                    padding="same",
                                    activation="relu")(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(padding="same")(x)
        x = self.mbconv(x, num_channels=32, output_channels=16, kernel_size=3)
        x = self.mbconv(x, num_channels=16, output_channels=24, kernel_size=3)
        x = tf.keras.layers.MaxPool2D(padding="same")(x)
        x = self.mbconv(x, num_channels=24, output_channels=24, kernel_size=3)
        x = self.mbconv(x, num_channels=24, output_channels=40, kernel_size=5)
        x = tf.keras.layers.MaxPool2D(padding="same")(x)
        x = self.mbconv(x, num_channels=40, output_channels=40, kernel_size=5)
        x = self.mbconv(x, num_channels=40, output_channels=80, kernel_size=3)
        for _ in range(2):
            x = self.mbconv(x, num_channels=80, output_channels=80, kernel_size=3)
        x = self.mbconv(x, num_channels=80, output_channels=112, kernel_size=5)
        x = tf.keras.layers.MaxPool2D(padding="same")(x)
        for _ in range(2):
            x = self.mbconv(x, num_channels=112, output_channels=112, kernel_size=5)
        x = self.mbconv(x, num_channels=112, output_channels=192, kernel_size=5)
        x = tf.keras.layers.MaxPool2D(padding="same")(x)
        for _ in range(3):
            x = self.mbconv(x, num_channels=192, output_channels=192, kernel_size=5)
        x = self.mbconv(x, num_channels=192, output_channels=320, kernel_size=3)
        return x

    def mbconv(self, inputs, num_channels, output_channels, kernel_size):
        x = tf.keras.layers.Conv2D(filters=num_channels,
                                    kernel_size=1,
                                    padding="same",
                                    activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(filters=num_channels,
                                    kernel_size=kernel_size,
                                    padding="same",
                                    activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1, padding="same")(x)
        x = tf.keras.layers.Add()([x, inputs])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(1, 1), padding="same")(x)
        return x


model = EfficientNetB0()
print(model(tf.zeros([1, 224, 224, 3])))