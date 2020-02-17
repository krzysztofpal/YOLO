import tensorflow as tf
import numpy as np
import PIL
import image_helper
import loss_functions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

TRAIN_FOLDER = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-3\\WIDER_train\\images\\'
TRAIN_METADATA_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-3\\wider_face_split\\wider_face_split\\wider_face_train_bbx_gt.txt'
VALIDATION_FOLDER = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-3\\WIDER_val\\WIDER_val\\images\\'
VALIDATION_METADATA_PATH = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-3\\wider_face_split\\wider_face_split\\wider_face_val_bbx_gt.txt'

GRIDS = 7

class box():
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    def fill(self, text_line):
        parts = text_line.split(' ')
        self.x1 = int(parts[0])
        self.y1 = int(parts[1])
        self.x2 = self.x1 + int(parts[2])
        self.y2 = self.y1 + int(parts[3])

    def data(self, image_width, image_height):
        x1 = self.x1 / image_width
        y1 = self.y1 / image_height
        x2 = self.x2 / image_width
        y2 = self.y2 / image_height

        return (x1,y1,x2,y2)

    def to_array(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])
        

class image():

    name = ""
    faces_count = 0
    
    def __init__(self, is_validation=False):
        self.boxes = []
        self.is_validation = is_validation

    def path(self):
        a = TRAIN_FOLDER if self.is_validation == False else VALIDATION_FOLDER
        return a + self.name

    def fill(self, text_lines):
        self.boxes = []
        self.name = text_lines[0].replace('\n', '')
        self.faces_count = int(text_lines[1].replace('\n', ''))
        for i in range(2, len(text_lines)):
            b = box()
            b.fill(text_lines[i])
            self.boxes.append(b)

    def show_predictions(self, model):
        path = self.path()
        im = PIL.Image.open(path)
        width, height = im.size
        
        xx,yy,zz = model.shape
        for x in range(xx):
            for y in range(yy):

                if(model[x,y,0] > 0.5):

                    cx = model[x,y,1] * width
                    cy = model[x,y,2] * height
                    minor = model[x,y,3] * width
                    major = model[x,y,4] * height

                    x1 = cx - minor/2
                    x2 = cx + minor/2
                    y1 = cy - major/2
                    y2 = cy + major/2

                    image_helper.draw_rectangle(im, (x1,y1,x2,y2), width=1, outline='red')
        
        im.show()


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, dim=(224,224,3), shuffle=True, rnd_rescale=True, rnd_multiply=True, rnd_color=True, rnd_crop=True, rnd_flip=True, debug=False, is_validation=False):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.rnd_rescale = rnd_rescale
        self.rnd_multiply = rnd_multiply
        self.rnd_color = rnd_color
        self.rnd_crop = rnd_crop
        self.rnd_flip = rnd_flip
        self.debug = debug
        self.is_validation = is_validation

        self.images = []  

        with open(TRAIN_METADATA_PATH if is_validation == False else VALIDATION_METADATA_PATH) as f:

            batch = []
            current_line = 0
            lines_to_read = 3
            for l in f:
                if(current_line == 1):
                    lines_to_read = max(int(l) + 2, 3)
                if current_line < lines_to_read:
                    batch.append(l)
                if(current_line == lines_to_read - 1):
                    im = image(self.is_validation)
                    im.fill(batch)
                    self.images.append(im)
                    batch = []
                    current_line = 0
                    lines_to_read = 3
                else:
                    current_line = current_line + 1
    
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        images = self.images[index * self.batch_size : (index + 1) * self.batch_size]

        X = np.empty((self.batch_size, *self.dim))
        Y = np.zeros((self.batch_size, GRIDS, GRIDS, 5), dtype=np.float32)
        entry = 0

        for i in images:

            with PIL.Image.open(i.path()) as img:
                boxes = np.zeros((len(i.boxes), 4))
                for a in range(len(i.boxes)):
                    boxes[a] = i.boxes[a].to_array()

                if(self.rnd_rescale):
                    old_width = img.width
                    old_height = img.height

                    

                    rescale = np.random.uniform(low=0.6, high=1.4)
                    new_width = int(old_width * rescale)
                    new_height = int(old_height * rescale)

                    img = img.resize((new_width, new_height))

                    for x in boxes:
                        x[0] *= new_width / old_width
                        x[1] *= new_height / old_height
                        x[2] *= new_width / old_width
                        x[3] *= new_height / old_height

                if(self.rnd_crop):

                    start_x = np.random.randint(0, high=np.floor(0.2 * img.width))
                    stop_x = img.width - np.random.randint(0, high=np.floor(0.2 * img.width))
                    start_y = np.random.randint(0, high=np.floor(0.2 * img.height))
                    stop_y = img.height - np.random.randint(0, high=np.floor(0.2 * img.height))

                    img = img.crop((start_x, start_y, stop_x, stop_y))

                    for x in boxes:
                        x[0] = max(x[0] - start_x, 0)
                        x[1] = max(x[1] - start_y, 0)
                        x[2] = min(x[2] - start_x, img.width)
                        x[3] = min(x[3] - start_y, img.height)
                    

                if(self.rnd_flip):
                    elem = np.random.choice([0, 90, 180, 270])

                    for x in boxes:

                        _x = x[0]- img.width / 2
                        _y = x[1]- img.height / 2
                        __x =x[2] - img.width / 2
                        __y =x[3] - img.height / 2

                        x[0] = img.width / 2 + _x * np.cos(np.deg2rad(elem)) - _y * np.sin(np.deg2rad(elem))
                        x[1] = img.height / 2 + _x * np.sin(np.deg2rad(elem)) + _y * np.cos(np.deg2rad(elem))
                        x[2] = img.width / 2 + __x * np.cos(np.deg2rad(elem)) - __y * np.sin(np.deg2rad(elem))
                        x[3] = img.height / 2 + __x * np.sin(np.deg2rad(elem)) + __y * np.cos(np.deg2rad(elem))

                    img = img.rotate(-elem)

                _wimg = img.width
                _himg = img.height

                # FRESHLY ADDED
                for x in boxes:
                    
                    x0 = x[0]
                    y0 = x[1]
                    x1 = x[2]
                    y1 = x[3]

                    tmp = x0
                    x0 = min(x0, x1)
                    x1 = max(tmp, x1)

                    tmp = y0
                    y0 = min(y0, y1)
                    y1 = max(tmp, y1)

                    x0 = max(x0, 0)
                    y0 = max(y0, 0)

                    y0 = min(y0, _himg)
                    x0 = min(x0, _wimg)
                    y1 = min(y1, _himg)
                    x1 = min(x1, _wimg)


                    x[0] = x0
                    x[1] = y0
                    x[2] = x1
                    x[3] = y1

                if(self.rnd_color):
                    enchancer = PIL.ImageEnhance.Color(img)
                    img = enchancer.enhance(np.random.uniform(low=0.5, high=1.5))

                    enhancer2 = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(np.random.uniform(low=0.7, high=1.3))

                img = img.resize((self.dim[0], self.dim[1]))
                img = img.convert('RGB')
                img = np.array(img)             

                X[entry] = preprocess_input(img.copy())
                
                for x in boxes:
                    
                    x1 = x[0]
                    y1 = x[1]
                    x2 = x[2]
                    y2 = x[3]

                    xc = GRIDS / _wimg * (x1 + (x2 - x1) / 2)
                    yc = GRIDS / _himg * (y1 + (y2 - y1) / 2)

                    fxc = int(xc)
                    if(fxc == GRIDS):
                        fxc = fxc - 1
                    fyc = int(yc)
                    if(fyc == GRIDS):
                        fyc = fyc - 1

                    Y[(entry, fyc, fxc, 0)] = (y2 - y1) / _himg
                    Y[(entry, fyc, fxc, 1)] = (x2 - x1) / _wimg
                    Y[(entry, fyc, fxc, 2)] = yc - fyc
                    Y[(entry, fyc, fxc, 3)] = xc - fxc
                    Y[(entry, fyc, fxc, 4)] = 1

                entry = entry + 1

        return X, Y


def display_batch(index=0, batch_size=32):
    d = DataGenerator(batch_size=batch_size, rnd_color=False, rnd_crop=True, rnd_flip=False, rnd_multiply=False, rnd_rescale=False)
    a = d.__getitem__(index)
    for i in range(batch_size):
        im = a[0][i]
        im = tf.cast(tf.math.multiply(tf.math.add(im, -tf.reduce_min(im)), 127.5), tf.int8)
        model = a[1][i]
        img = PIL.Image.fromarray(np.asarray(im), 'RGB')
        width = img.width
        height = img.height
        for row in range(model.shape[0]):
            for column in range(model.shape[1]):
                vector = model[row][column]
                exists = vector[4] > 0.5
                if(exists):
                    bh = vector[0] * height
                    bw = vector[1] * width
                    yc = (vector[2] + row) / GRIDS * height
                    xc = (vector[3] + column) / GRIDS * width

                    x1 = xc - bw / 2
                    x2 = xc + bw / 2
                    y1 = yc - bh / 2
                    y2 = yc + bh / 2

                    if(x1 > x2):
                        temp = x1
                        x1 = x2
                        x2 = temp

                    if(y1 > y2):
                        temp = y1
                        y1 = y2
                        y2 = temp

                    image_helper.draw_rectangle(img, (x1, y1, x2, y2), outline='green', width=1)

        img = img.resize((1000, 1000))
        img.show()

display_batch(batch_size=6)
