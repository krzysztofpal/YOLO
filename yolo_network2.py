import loss_functions
import numpy as np
import tensorflow as tf
import data_generator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.optimizers import SGD


CHECKPOINT_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\exported-networks\\localization_network_checkpoint-3.cktp'

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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(288,288,3), padding='same'),
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid', padding='same')
])

optimizer = SGD(lr=LEARNING_RATE, decay=LR_DECAY, momentum=0.9, nesterov=False)
model.compile(loss=loss_functions.detection_loss(), optimizer=optimizer, metrics=[])

train_datagen = data_generator.DataGenerator(dim=(288,288,3), rnd_color=False, rnd_crop=False, rnd_flip=False, rnd_multiply=False, rnd_rescale=False)


val_generator = data_generator.DataGenerator(dim=(288,288,3), rnd_rescale=False, rnd_multiply=False, rnd_crop=False, rnd_flip=False, debug=False, is_validation=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

model.fit_generator(
    generator=train_datagen,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    callbacks=[cp_callback],
    validation_data = val_generator
)