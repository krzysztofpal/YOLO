import loss_functions
import numpy as np
import tensorflow as tf
import data_generator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.optimizers import SGD
import validation_callback
import yolo_params
import yolo_models


CHECKPOINT_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\exported-networks\\yolo_from_scratch.cktp'

LPHA = 0.35

GRID_SIZE = yolo_params.grid_size()
IMAGE_SIZE = yolo_params.image_size()

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




model = yolo_models.model_from_scratch()


optimizer = SGD(lr=LEARNING_RATE, decay=LR_DECAY, momentum=0.9, nesterov=False)
model.compile(loss=loss_functions.detection_loss(), optimizer=optimizer, metrics=[])

train_datagen = data_generator.DataGenerator(batch_size=BATCH_SIZE, dim=(IMAGE_SIZE, IMAGE_SIZE, 3), rnd_color=True, rnd_crop=True, rnd_flip=False, rnd_multiply=True, rnd_rescale=True)
#train_datagen = data_generator.DataGenerator(batch_size=BATCH_SIZE, dim=(IMAGE_SIZE, IMAGE_SIZE, 3), rnd_color=False, rnd_crop=False, rnd_flip=False, rnd_multiply=False, rnd_rescale=False)


#model.load_weights(CHECKPOINT_PATH)

val_generator = data_generator.DataGenerator(batch_size=BATCH_SIZE, dim=(IMAGE_SIZE, IMAGE_SIZE, 3), rnd_rescale=False, rnd_multiply=False, rnd_crop=False, rnd_flip=False, debug=False, is_validation=True)
validation_callback = validation_callback.Validation(generator=val_generator)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)
stop = tf.keras.callbacks.EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou", factor=0.6, patience=5, min_lr=1e-6, verbose=1, mode="max")

model.fit_generator(
    generator=train_datagen,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    callbacks=[cp_callback, validation_callback, stop, reduce_lr],
)