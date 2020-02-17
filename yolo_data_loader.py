import glob
import numpy
from PIL import Image, ImageDraw
import image_helper

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
GRIDS_PER_ROW = 7
GRIDS_PER_COLUMN = 7

metadata_path = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-1\\FDDB-folds\\FDDB-folds'
images_root_path = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-1\\originalPics'


class ellipse_metadata:
    major_axis_radius = 0
    minor_axis_radius = 0
    angle = 0
    center_x = 0
    center_y = 0
    detection_score = 0

    def box(self):

        cx = self.center_x
        cy = self.center_y
        major = self.major_axis_radius
        minor = self.minor_axis_radius
        angle = self.angle

        ux = minor * numpy.cos(angle)
        uy = minor * numpy.sin(angle)
        vx = major * numpy.cos(angle + numpy.pi / 2)
        vy = major * numpy.sin(angle + numpy.pi / 2)

        b_hh = numpy.sqrt(ux ** 2 + vx ** 2)
        b_hw = numpy.sqrt(uy ** 2 + vy ** 2)

        x1 = cx - b_hw
        y1 = cy - b_hh
        x2 = cx + b_hw
        y2 = cy + b_hh

        box = (x1, y1, x2, y2)
        
        return box

    def data(self, image_width, image_height):
        x = self.center_x / image_width
        y = self.center_y / image_height
        minor = self.minor_axis_radius / image_width
        major = self.major_axis_radius / image_height
        angle = (self.angle + numpy.pi) / numpy.pi / 2

        arr = numpy.array([x,y,minor,major,angle])
        return arr

    def fill(self, image_width, image_height, data):
        self.center_x = data[0] * image_width
        self.center_y = data[1] * image_height
        self.minor_axis_radius = data[2] * image_width
        self.major_axis_radius = data[3] * image_height
        self.angle = data[4] * numpy.pi * 2 - numpy.pi

    def data_yolo(self, image_width, image_height):
        
        x = self.center_x / image_width
        y = self.center_y / image_height
        minor = self.minor_axis_radius / image_width * 2
        major = self.major_axis_radius / image_height * 2

        d = numpy.array([x,y,minor,major])

        row = int(d[1] * GRIDS_PER_ROW)
        if(row == GRIDS_PER_ROW):
            row = row - 1
        column = int(d[0] * GRIDS_PER_COLUMN)
        if(column == GRIDS_PER_COLUMN):
            column = column - 1
        return (row, column, d)

class image_metadata:
    image_name = ""
    faces_count = 0
    ellipses = []

    def path(self):
        path = images_root_path + '\\' + self.image_name + '.jpg'
        return path

    def show(self):
        path = self.path()
        im = Image.open(path)
        for i in self.ellipses:
            box = i.box()
            image_helper.draw_ellipse(im, box, width=1)
        im.show()

    def tensor(self):
        path = self.path()
        im = Image.open(path)
        original_width, original_height = im.size
        im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        arr = numpy.zeros([height, width, 3])
        for i in range(height):
            for j in range(width):
                
                pixel = pixels[i][j]
                
                if isinstance(pixel, int):
                    pixel = [pixel, pixel, pixel]

                for k in range(3):
                    arr[i,j,k] = pixel[k]
        
        return (arr, original_width, original_height)

    def data_localization(self):

        arr, original_width, original_height = self.tensor()
        
        ellipse = self.ellipses[0].data(original_width, original_height)

        return (arr, ellipse)

    def data_yolo(self):

        arr, original_width, original_height = self.tensor()

        label = numpy.zeros([GRIDS_PER_ROW, GRIDS_PER_COLUMN, 5])

        for i in self.ellipses:
            row, column, data = i.data_yolo(original_width, original_height)
            label[row, column, 4] = 1
            label[row, column, 0] = data[0]
            label[row, column, 1] = data[1]
            label[row, column, 2] = data[2]
            label[row, column, 3] = data[3]

        return (arr, label)


    def eval(self, model):
        path = self.path()
        im = Image.open(path)
        width, height = im.size
        for i in self.ellipses:
            box = i.box()
            image_helper.draw_ellipse(im, box, width=1)
        
        ne = ellipse_metadata()
        ne.fill(width, height, model)
        ne_box = ne.box()

        image_helper.draw_ellipse(im, ne_box, width=2, outline='red')

        im.show()

    def eval_yolo(self, model):
        path = self.path()
        im = Image.open(path)
        width, height = im.size
        for i in self.ellipses:
            box = i.box()
            image_helper.draw_ellipse(im, box, width=1)
        
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


def load_yolo_data():

    metadata = []

    metadata_names = glob.glob(metadata_path + '\\*-ellipseList.txt')

    for i in metadata_names:
        
        batches = []
        batch = []
        f = open(i)

        current_line = 0
        lines_to_read = 3
        for l in f:
            if(current_line == 1):
                lines_to_read = int(l) + 2
            if current_line < lines_to_read:
                batch.append(l)
            if(current_line == lines_to_read - 1):
                batches.append(batch)
                batch = []
                current_line = 0
                lines_to_read = 3
            else:
                current_line = current_line + 1
                
        f.close()

        for b in batches:
            name = b[0].replace('\n','')
            ellipses = []
            for e in range(2, len(b)):
                line = b[e]
                pieces = line.split(" ")
                el = ellipse_metadata()
                el.major_axis_radius = float(pieces[0])
                el.minor_axis_radius = float(pieces[1])
                el.angle = float(pieces[2])
                el.center_x = float(pieces[3])
                el.center_y = float(pieces[4])
                el.detection_score = float(pieces[6])
                ellipses.append(el)
            
            meta = image_metadata()
            meta.image_name = name
            meta.faces_count = len(ellipses)
            meta.ellipses = ellipses
            metadata.append(meta)

    return metadata

def evaluate(index=-1):
    metadata = load_yolo_data()
    one_person_set = metadata

    p = one_person_set[index]

    return p

def save_localization_data():

    path = 'C:\\Users\\Admin\\Desktop\\face-recognition\\datasets\\dataset-1\\extracted\\yolo224'

    metadata = load_yolo_data()
    one_person_set = metadata #list(filter(lambda x: (x.faces_count == 1), metadata))
    
    images = []
    labels = []
    
    _all = len(one_person_set)
    i = 0

    while(len(one_person_set) > 0):
        p = one_person_set[0].data_yolo()
        one_person_set.pop()
        images.append(p[0])
        labels.append(p[1])
        i = i + 1
        print("Extracted " + str(i) + ' / ' + str(_all) + ' files')
        

    l = len(images)
    _l = int(l * 0.8)

    train_images = images[0:_l]
    train_labels = labels[0:_l]
    test_images = images[_l:l]
    test_labels = labels[_l:l]

    print("Files ready. Saving time")
    numpy.savez_compressed(path, train_images = train_images, train_labels = train_labels, test_images = test_images, test_labels = test_labels)
