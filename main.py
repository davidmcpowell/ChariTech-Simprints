import numpy as np
import png
import os
from sklearn import tree
import random
from scipy import ndimage as ndi
from skimage import feature
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation


def construct_nn():
    model = Sequential()
    model.add(Dense(units=64, input_dim=32))
    model.add(Activation('tanh'))
    model.add(Dense(units=5))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
    return model


def classify(method='tree'):
    classifier, predictions, test_labels = train(method)
    results = compare(test_labels, predictions)
    print results
    
def compare(labels, predicted_labels):
    correct = sum((1 for i in xrange(len(labels))
                     if labels[i] == predicted_labels[i]))
    return correct / float(len(labels))

# this function does stuff
def train(method='tree'):
    image_labels = get_image_labels()

    training_data, training_images, test_data, test_images = get_training_and_test_data()
    train_labels = images_to_labels(training_images, image_labels)
    test_labels = images_to_labels(test_images, image_labels)
    print 'Extracted Training Data'
    if method == 'tree':
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(training_data, train_labels)
        predictions = classifier.predict(test_data)
    else:
        classifier = construct_nn()
        nn_labels = get_nn_labels(train_labels)
        classifier.fit(training_data, nn_labels)
        predictions =  classifier.predict(test_data)
        predictions = format_predictions(predictions)
    
    return classifier, predictions, test_labels

def format_predictions(predictions):
    new_predictions = []
    for prediction in predictions:
        new_predictions.append(np.argmax(prediction))
    print new_predictions
    return new_predictions


def get_nn_labels(labels):
    nn_labels = np.zeros((len(labels), 5))
    for i in xrange(len(labels)):
        value = labels[i]
        nn_labels[i, value] = 1
    return nn_labels 

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
                        if random.random() > 0.8]
    print 'Number of images used:', len(image_files)
    return image_files

def get_training_and_test_data():
    image_files = get_all_image_file_names()
    np.random.shuffle(image_files)
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
            image[row_num, :] = np.array(row)
            row_num += 1
        # compressed = compress(image)
        # compressed = image.flatten()
        # compressed = convert_to_binary(image).flatten()
        compressed = get_features(image, 128)
    return compressed

def compress(values):
    grid_size = 1
    num_vals = 512 / grid_size
    compressed_vals = np.zeros((num_vals, 512/grid_size))
    for i in xrange(num_vals):
        for j in xrange(num_vals):
            small_grid = values[grid_size*i : (i+1)*grid_size, grid_size*j: grid_size*(j+1)]
            compressed_vals[i, j] = np.mean(small_grid)
    compressed_vals = compressed_vals.flatten()
    return compressed_vals

def convert_to_binary(image):
    width = len(image)
    bin_image = np.zeros((width, width))
    for i in xrange(width):
        for j in xrange(width):
            bin_image[i, j] = (image[i, j] > 144)

    # for row in bin_image:
    #     print row
    return bin_image


def plot_bin_image(image):
    image = image.astype(int)
    file_name = '144.png' % (threshold)
    plt.imsave(file_name, np.array(image), cmap=cm.gray)

def get_edges(image):
    edges = feature.canny(image, sigma=3)
    # a = 0
    # for row in edges:
    #     for pixel in row:
    #         if pixel:
    #             a += 1
    return edges

def get_features(image, grid_size):
    num_vals = 512 / grid_size
    vertical = np.zeros((num_vals, num_vals))
    horizontal = np.zeros((num_vals, num_vals))
    for i in xrange(num_vals):
        for j in xrange(num_vals):
            small_grid = image[grid_size*i : (i+1)*grid_size, grid_size*j: grid_size*(j+1)]
            vertical[i, j], horizontal[i, j] = calculate_gradient(small_grid)
    features = np.concatenate([vertical.flatten(), horizontal.flatten()])
    return features

def calculate_gradient(image_section):
    vertical = 0
    horizontal = 0
    num_rows = len(image_section)
    num_cols = len(image_section[0])
    for row in image_section:
        for i in xrange(num_rows-1):
            horizontal += abs(row[i] - row[i+1])

    for column in image_section.T:
        for i in xrange(len(column)-1):
            vertical += abs(column[i] - column[i+1])
    vertical = vertical/float(((num_rows - 1)*(num_cols-1)))
    horizontal = horizontal/float(((num_rows - 1)*(num_cols-1)))
    return horizontal, vertical



if __name__ == '__main__':
    # for threshold in xrange (128, 164, 4):
    #     image = read_png_file('fingerprintClassification/trainingSet/A/f1675_02.png')
    #     image = convert_to_binary(image, threshold)
    #     # # image = get_edges(image)
    #     plot_bin_image(image, threshold)
    # plot_bin_image(np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]]))
    classify()



