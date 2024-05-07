import os
import pathlib
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed """
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import tensorflow as tf
from keras import callbacks

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# download fashion mnist dataset and split into train and test sets
#fashion_mnist = tf.keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# create a list of the labels from the data
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               #'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# shows the number of images and their resolution and num of training labels
#print(train_images.shape)
#print(len(train_labels))

# scale images between 0 to 1
#train_images = train_images / 255.0
#test_images = test_images / 255.0



# autotune image
#AUTOTUNE = tf.data.AUTOTUNE

# windows file path
#dataset_url = "file://C:/Users/kevdi/Desktop/UA Files/Applied Machine Learning/Project 3/chess boards/"

# wsl file path
#dataset_url = "file:///mnt/c/Users/kevdi/Desktop/UA Files/Applied Machine Learning/Project 3/chess boards.zip"
datasetDirectoryPath = "../chess boards/individualTrain"
#archive = tf.keras.utils.get_file(origin = dataset_url, extract = True)
#data_directory = pathlib.Path(datasetDirectoryPath)


#image_count = len(list(data_directory.glob('*/*.jpeg')))
#print("num images:", image_count)
print("Creating dataset")

train_images_ds = tf.keras.utils.image_dataset_from_directory(
    datasetDirectoryPath,
    #validation_split = .2,
    #subset = "training",
    seed = 1337,
    image_size = (25, 25), # resizes images to specified size
    shuffle = True, # sorted alphanumerically otherwise
    #batch_size = 32, # default value
    labels = "inferred",
    color_mode = "grayscale",
)

#AUTOTUNE = tf.data.AUTOTUNE
#train_images_ds.cache().prefetch(buffer_size=AUTOTUNE)
#train_images_ds = tf.data.Dataset.load('./dataSets/traindata')
print("dataset Created")
# normalize the gray scale values to values between 0 and 1 (floats)
#normalization_layer = tf.keras.layers.Rescaling(1./255)
#normalized_train_ds = train_images_ds.map()

#print("Saving dataset...")
#tf.data.Dataset.save(train_images_ds, './datatSets/traindata')
#print("dataset saved")
#test_images = test_images.cache().prefetch(buffer_size=AUTOTUNE)

try:
    model = tf.keras.models.load_model('./ChessModel2Save.keras')
    print("Model loaded.\n")
except:
    print("No saved model found.\n")
    model = False

runTrain = True

if not model or not runTrain:
    model = tf.keras.Sequential()
    #model = tf.keras.Model()
    model.add(tf.keras.layers.Input(shape=(25,25)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

    history = model.fit(train_images_ds, epochs=25, callbacks=[earlystopping], validation_split=.8)

print("history:", history)

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print('\nTest accuracy:', test_acc)

model.save('./ChessModel2Save.keras')
