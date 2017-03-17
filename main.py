import numpy as np
import png
import os

def classify():
    pass

# this function does stuff
def train():
    image_labels = get_image_labels()


def get_image_labels():
    outer_dir =  'fingerprintClassification/trainingSet/'
    dict_fingerprints = {}
    labels = ['A', 'L', 'R', 'T', 'W']
    for label in labels:
        image_dir = os.path.join(outer_dir, label)
        file_names = os.listdir(image_dir)
        
        for file_name in file_names:
            dict_fingerprints[file_name] = 'image' + label

    return dict_fingerprints


def read_png_file(file_name):
    with open(file_name, 'r') as f:
        r = png.Reader(f)
        width, height, values, properties = r.read()
        image = np.zeros((width, height))
        row_num = 0
        for row in values:
            image[row_num, :] = np.array(row[1])
            row_num += 1
        compressed = compress(image)

def compress(values):

    return compressed_vals

if __name__ == '__main__':
    read_png_file('fingerprintClassification/trainingSet/A/f1675_02.png')
   



