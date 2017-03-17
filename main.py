import numpy as np
import png
import os
from sklearn import tree

def classify():
    clf, X, labels = train()
    predicted_labels = clf.predict(X)
    results = compare(labels, predicted_labels)
    print results
    
def compare(labels, predicted_labels):
    correct = sum((1 for i in xrange(len(actual))
                     if labels[i] == predicted_labels[i]))
    return correct / float(len(actual))

# this function does stuff
def train():
    image_labels = get_image_labels()
    classifier = tree.DecisionTreeClassifier()
    X, image_files = get_training_data()
    #extracts file name from relative path
    file_names = [image_file.split('/')[-1] for image_file in image_files]
    labels = [image_labels[file_name] for file_name in file_names]
    clf.fit(X, labels)
    return clf, X, labels

def get_all_image_file_names():
    outer_dir =  'fingerprintClassification/trainingSet/'
    labels = ['A', 'L', 'R', 'T', 'W']
    image_files = [os.path.join(outer_dir, label, file_name)
                        for label in labels
                        for file_name in os.listdir(os.path.join(outer_dir, label))]
    return image_files

def get_training_data():
    image_files = get_all_image_file_names()
    data = np.array([read_png_file(image_file) for image_file in image_files])
    return data, image_files


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
    return compressed

def compress(values):
    compressed_vals = []
    return compressed_vals

if __name__ == '__main__':
    # read_png_file('fingerprintClassification/trainingSet/A/f1675_02.png')
    print get_training_data()
   



