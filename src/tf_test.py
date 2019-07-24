# ASSUMES generate_data.py has ran!

# tensorflow sandbox
import numpy as np
import pandas as pd
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# read labels
# path to file that will contain camera properties of generated images
params_filename = "camera_properties.csv"
train_dir = os.path.join(os.path.dirname(__file__), '../training_data')
params_path = os.path.join(train_dir, params_filename)
output_df = pd.read_csv(params_path)
# convert to numpy array
output = output_df.to_numpy()
# drop first column (idx)
output = output[:, 1:]

# dummy values to be populated later
images = None
image_set_initialized = False
w = -1
h = -1
num_images = -1

# read images
print('Reading Images...')
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith(".jpg"):
            # found an image in the train directory
            # format is <idx>.jpg
            idx_str = file[:-4]
            # get as an integer, this is the corresponding index/row of camera_properties.csv
            idx = int(idx_str)

            # read image
            im_path = os.path.join(root, file)
            im = cv2.imread(im_path, 0)  # 0 flag => grayscale

            # initialize train set if necessary (allocate memory, find size of images, etc.)
            if (not image_set_initialized):
                # assumes every image in train_data/. is an image except the camera_properties.csv files
                num_images = len(files) - 1
                (w, h) = im.shape
                images = np.zeros((num_images, h, w, 1), np.uint8)
                image_set_initialized = True

            # store image in images
            if (image_set_initialized):
                images[idx, :, :, 0] = im/255.0  # normalize

# partition data into train/test data
# proportion of data that will be used for train versus testing
train_ratio = 0.8
# split images/output into train/testing data
train_num_images = int(num_images * train_ratio)
test_num_images = num_images - train_num_images
# randomly permute/shuffle images and output
idx = list(range(0, num_images))
np.random.shuffle(idx)
train_idx = idx[:train_num_images]
test_idx = idx[train_num_images:]
# get images/output associated with train_idx/test_idx
train_images = images[train_idx, :, :, :]
test_images = images[test_idx, :, :, :]
train_output = output[train_idx, :]
test_output = output[test_idx, :]

# build neural network, REVISIT - structure of network is more or less random
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(w, h, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='linear'))
# REVISIT loss function is important, mean_squared_error may not be reasonable
model.compile(optimizer='adam',
              loss='mean_squared_error')
# train model
model.fit(train_images, train_output, epochs=5)

# evaluate model on test data
test_loss = model.evaluate(test_images, test_output)
print('Test Loss: ' + str(test_loss))
