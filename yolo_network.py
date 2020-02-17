import loss_functions
import numpy as np
import tensorflow as tf
import data_generator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.optimizers import SGD


CHECKPOINT_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\exported-networks\\localization_network_checkpoint_ni_data_augemented.cktp'

LPHA = 0.35

GRID_SIZE = 7
IMAGE_SIZE = 224

# first train with frozen weights, then fine tune
TRAINABLE = False
WEIGHTS = "model-0.64.h5"

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0005
LR_DECAY = 0.0001

MULTITHREADING = False
THREADS = 1




model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, alpha=0.35, weights="imagenet")
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
regularizer = l2(0.0005 / 2)
for weight in model.trainable_weights:
    with tf.keras.backend.name_scope("weight_regularizer"):
        model.add_loss(lambda: regularizer(weight)) # in tf2.0: lambda: regularizer(weight)


optimizer = SGD(lr=LEARNING_RATE, decay=LR_DECAY, momentum=0.9, nesterov=False)
model.compile(loss=loss_functions.detection_loss(), optimizer=optimizer, metrics=[])

train_datagen = data_generator.DataGenerator(rnd_color=False, rnd_crop=False, rnd_flip=False, rnd_multiply=False, rnd_rescale=False)

#model.load_weights(CHECKPOINT_PATH)

val_generator = data_generator.DataGenerator(rnd_rescale=False, rnd_multiply=False, rnd_crop=False, rnd_flip=False, debug=False, is_validation=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

model.fit_generator(
    generator=train_datagen,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    callbacks=[cp_callback],
    validation_data = val_generator
)