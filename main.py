import numpy as np
import png
import os

def classify():
    pass

# this function does stuff
def train():
    image_labels = get_image_labels()

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
    print compressed

def compress(values):
    num_vals = 512 /16
    compressed_vals = np.zeros((num_vals, 512/16))
    for i in xrange(num_vals):
        for j in xrange(num_vals):
            small_grid = values[16*i : i*16+16, 16*j: 16*j +16]
            compressed_vals[i, j] = np.mean(small_grid)

    return compressed_vals

if __name__ == '__main__':
    read_png_file('fingerprintClassification/trainingSet/A/f1675_02.png')

