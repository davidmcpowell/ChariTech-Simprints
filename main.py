import numpy as np
import png

def classify():
    pass


def read_png_file(file_name):
    with open(file_name, 'r') as f:
        r = png.Reader(f)
        print r.read()


if __name__ == '__main__':
    read_png_file('fingerprintClassification/trainingSet/A/f1675_02.png')

