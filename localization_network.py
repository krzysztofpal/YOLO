import loss_functions
import numpy as np
import tensorflow as tf
import yolo_data_loader
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

DATA_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-1\\extracted\\yolo224.npz'
CHECKPOINT_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\exported-networks\\localization_network_checkpoint.cktp'
EXPORT_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\exported-networks\\localization_network_yolo-temp.h5'
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

with np.load(DATA_PATH) as data:
    train_examples = data["train_images"] / 255.0
    train_labels = data["train_labels"]
    test_examples = data["test_images"] / 255.0
    test_labels = data["test_labels"]

#train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
#test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

#train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

train_examples = train_examples.reshape(2276, 224, 224, 3)
test_examples = test_examples.reshape(569, 224, 224, 3)

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

#model = tf.keras.models.Sequential([
#    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(288,288,3), padding='same'),
#    tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D(2,2),
#    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D(2,2),
#    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D(2,2),
#    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D(2,2),
#    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D(2,2),
#    tf.keras.layers.Conv2D(192, (3,3), activation='relu'),
#    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),
#    tf.keras.layers.Conv2D(5, (1,1), activation='sigmoid', padding='same')
#])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

opt=tf.keras.optimizers.SGD(lr=8e-7, momentum=0.9)
model.compile(optimizer=opt, loss=loss_functions.detection_loss(), metrics=['mae', 'mse'])

#model.load_weights(CHECKPOINT_PATH)

model.fit(train_examples, train_labels, epochs=200, validation_data=(test_examples, test_labels)) 

model.save(EXPORT_PATH)

img = yolo_data_loader.evaluate(index=-1)
data = img.data_yolo()
img_pixels = data[0] / 255.0
img_pixels = np.array([img_pixels])
predict = model.predict(img_pixels)
predict = predict[0]

print("Prediction: ")
print(predict)
print("Requested: ")
print(data[1])

loss = tf.keras.losses.MSE(data[1], predict)
print(loss)

img.eval_yolo(predict)




