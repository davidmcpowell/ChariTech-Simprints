import numpy as np
import png
import os
from sklearn import tree
import random

def classify():
    classifier, test_data, test_labels = train()
    predicted_labels = classifier.predict(test_data)
    results = compare(test_labels, predicted_labels)
    print results
    
def compare(labels, predicted_labels):
    correct = sum((1 for i in xrange(len(labels))
                     if labels[i] == predicted_labels[i]))
    return correct / float(len(labels))

# this function does stuff
def train():
    image_labels = get_image_labels()
    classifier = tree.DecisionTreeClassifier()
    training_data, training_images, test_data, test_images = get_training_and_test_data()
    #extracts file name from relative path
    train_labels = images_to_labels(training_images, image_labels)
    test_labels = images_to_labels(test_images, image_labels)
    classifier.fit(training_data, train_labels)
    return classifier, test_data, test_labels

def images_to_labels(image_list, image_labels):
    file_names = [image_file.split('/')[-1] for image_file in image_list]
    labels = [image_labels[file_name] for file_name in file_names]
    label_converter = {'imageA' : 0, 'imageL' : 1, 'imageR' : 2, 'imageT' : 3, 'imageW' : 4}
    number_labels = [label_converter[label] for label in labels]
    return number_labels

def get_all_image_file_names():
    outer_dir =  'fingerprintClassification/trainingSet/'
    labels = ['A', 'L', 'R', 'T', 'W']
    image_files = [os.path.join(outer_dir, label, file_name)
                        for label in labels
                        for file_name in os.listdir(os.path.join(outer_dir, label))
                        if random.random() > 0]
    return image_files

def get_training_and_test_data():
    image_files = get_all_image_file_names()
    training_images = image_files[:len(image_files)/2]
    test_images = image_files[len(image_files)/2:]
    training_data = np.array([read_png_file(image_file) for image_file in training_images])
    test_data = np.array([read_png_file(image_file) for image_file in test_images])
    return training_data, training_images, test_data, test_images

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
    num_vals = 512 /16
    compressed_vals = np.zeros((num_vals, 512/16))
    for i in xrange(num_vals):
        for j in xrange(num_vals):
            small_grid = values[16*i : i*16+16, 16*j: 16*j +16]
            compressed_vals[i, j] = np.mean(small_grid)
    compressed_vals = compressed_vals.flatten()
    return compressed_vals

if __name__ == '__main__':
    classify()



