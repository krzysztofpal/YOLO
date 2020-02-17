import tensorflow as tf
import numpy as np
import yolo_data_loader

EXPORT_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\exported-networks\\localization_network_yolo.h5'


model = tf.keras.models.load_model(EXPORT_PATH)

for i in range(0,1):
    img = yolo_data_loader.evaluate(index=i)
    data = img.data_yolo()
    img_pixels = data[0] / 255.0
    img_pixels = np.array([img_pixels])
    predict = model.predict(img_pixels)
    predict = predict[0]

    print("Prediction: ")
    print(predict)
    print("Requested: ")
    print(data[1])

    loss = tf.keras.losses.MAE(data[1], predict)
    print(loss)

    img.eval(predict)