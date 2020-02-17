import tensorflow as tf
import numpy as np

def mobile_net_v2_model():
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), include_top=False, alpha=0.35, weights="imagenet")
    for layer in model.layers:
        layer.trainable = False

    block = model.get_layer("block_16_project_BN").output

    x = tf.keras.layers.Conv2D(112, padding="same", kernel_size=3, strides=1, activation="relu")(block)
    x = tf.keras.layers.Conv2D(112, padding="same", kernel_size=3, strides=1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(5, padding="same", kernel_size=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=model.input, outputs=x)

    # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
    # see https://arxiv.org/pdf/1711.05101.pdf
    regularizer = tf.keras.regularizers.l2(0.0005 / 2)
    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope("weight_regularizer"):
            model.add_loss(lambda: regularizer(weight)) # in tf2.0: lambda: regularizer(weight)

    return model

def model_from_scratch():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(288,288,3), padding='same'),
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid', padding='same')
    ])
    return model