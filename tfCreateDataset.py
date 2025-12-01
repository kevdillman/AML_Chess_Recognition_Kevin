import os, json, random
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# set random seeds
SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# set parameters for use in dataset
IMAGE_SIZE = (50, 50)
BATCH_SIZE = 32
NUM_EPOCHS = 20
AUTOTUNE = tf.data.experimental.AUTOTUNE
DIRECTORY_PATH = "../chess boards/largeImages"
DATA_DIR = Path(DIRECTORY_PATH)
SAVE_DIR = Path("./artifacts")
SAVE_DIR.mkdir(exist_ok=True)

# -----------------------------
# 3) Utility to read & preprocess images
# -----------------------------
def convertImagesToTensor(path, label, classNames, imageSize=IMAGE_SIZE):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)            # ensure 3 channels
    image = tf.image.resize(image, imageSize)
    image = tf.cast(image, tf.float32) / 255.0              # normalize to [0,1]
    return image, tf.one_hot(label, len(classNames))

# -----------------------------
# 4) Build tf.data datasets
# -----------------------------
def makeDataset(path, labels, classNames, training=False, batchSize=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((path, labels))

    # Data augmentation (applied in dataset pipeline or in-model)
    imageRandomizer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.05),])

    #if training:
        #ds = ds.shuffle(buffer_size=len(path), seed=SEED)

    ds = ds.map(lambda imagePath, imageLabel: convertImagesToTensor(imagePath, imageLabel, classNames), num_parallel_calls=AUTOTUNE)

    # apply augmentation on-the-fly
    if training:
        ds = ds.shuffle(buffer_size=len(path), seed=SEED)
        ds = ds.map(lambda imagePath, imageLabel: (imageRandomizer(imagePath, training=True), imageLabel), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batchSize).prefetch(AUTOTUNE)
    return ds

# load training, validation, and test data sets
try:
    print("\nTrying to load datasets...\n")

    with open(SAVE_DIR / "trainElementSpec.pkl", "rb") as trainElementSpecFile:
        trainElementSpec = pickle.load(trainElementSpecFile)
    trainDS = tf.data.experimental.load(str(SAVE_DIR / "traindata"), element_spec=trainElementSpec)

    with open(SAVE_DIR / "validationElementSpec.pkl", "rb") as validationElementSpecFile:
        validationElementSpec = pickle.load(validationElementSpecFile)
    validationDS = tf.data.experimental.load(str(SAVE_DIR / "validationdata"), element_spec=validationElementSpec)

    with open(SAVE_DIR / "testElementSpec.pkl", "rb") as testElementSpecFile:
        testElementSpec = pickle.load(testElementSpecFile)
    testDS = tf.data.experimental.load(str(SAVE_DIR / "testdata"), element_spec=testElementSpec)

    print("Loaded train, validation, and test datasets")

    with open(SAVE_DIR / "classWeights.json", "r") as classWeightsFile:
        trainClassWeightDictionary = json.load(classWeightsFile)
    # Ensure keys are ints (JSON stores them as strings)
    trainClassWeightDictionary = {int(key): float(value) for key, value in trainClassWeightDictionary.items()}

    print(f"Loaded class weights: {trainClassWeightDictionary}")

except:
    print("Error loading datasets, creating new...\n")
    # collect class labels from directories
    classNames = sorted([directory.name for directory in DATA_DIR.iterdir() if directory.is_dir()])
    print(f"Found classes: {classNames}\n")

    # build mapping: className -> index
    classIndex = {name: index for index, name in enumerate(classNames)}

    # Gather all files and label indices
    filePaths = []
    fileLabel = []
    for name in classNames:
        for imagePath in (DATA_DIR / name).glob("*.jpeg"):
            filePaths.append(str(imagePath))
            fileLabel.append(classIndex[name])

    filePaths = np.array(filePaths)
    fileLabel = np.array(fileLabel)
    print("Total images:", len(filePaths))

    # split train/test/validation (80/10/10)
    trainPaths, reservePaths, trainLabels, reserveLabels = train_test_split(
    filePaths,
    fileLabel,
    test_size=0.2, # the 20% will be split into test and validation subsets
    stratify=fileLabel,
    random_state=SEED)

    validationPaths, testPaths, validationLabels, testLabels = train_test_split(
    reservePaths,
    reserveLabels,
    test_size=0.5,
    stratify=reserveLabels,
    random_state=SEED)

    print(f"Train items:\t\t{len(trainPaths)}")
    print(f"Test items:\t\t{len(testPaths)}")
    print(f"Validation items:\t{len(validationPaths)}")

    # save class names mapping
    with open(SAVE_DIR / "classNames.json", "w") as classNameFile:
        json.dump(classNames, classNameFile)

    trainDS = makeDataset(trainPaths, trainLabels, classNames, training=True)
    tf.data.experimental.save(trainDS, str(SAVE_DIR / "traindata"))
    with open(SAVE_DIR / "trainElementSpec.pkl", "wb") as trainElementSpecFile:
        pickle.dump(trainDS.element_spec, trainElementSpecFile)

    validationDS   = makeDataset(validationPaths, validationLabels, classNames, training=False)
    tf.data.experimental.save(validationDS, str(SAVE_DIR / "validationdata"))
    with open(SAVE_DIR / "validationElementSpec.pkl", "wb") as validationElementSpecFile:
        pickle.dump(validationDS.element_spec, validationElementSpecFile)

    testDS  = makeDataset(testPaths, testLabels, classNames, training=False)
    tf.data.experimental.save(testDS, str(SAVE_DIR / "testdata"))
    with open(SAVE_DIR / "testElementSpec.pkl", "wb") as testElementSpecFile:
        pickle.dump(testDS.element_spec, testElementSpecFile)

    # calculate class weights of training data set
    trainClassWeights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(trainLabels),
        y=trainLabels)
    trainClassWeightDictionary = {i: float(w) for i, w in enumerate(trainClassWeights)}

    with open(SAVE_DIR / "classWeights.json", "w") as classWeightsFile:
        json.dump(trainClassWeightDictionary, classWeightsFile, indent = 4)
    print(f"Saved class weights: {trainClassWeightDictionary}")


#print(f"\nClass weights: {trainClassWeightDictionary}\n")
# -----------------------------
# 5) Compute class weights (optional, if imbalance)
# -----------------------------
"""trainClassWeights = compute_class_weight(
    classWeight='balanced',
    classes=np.unique(trainLabels),
    y=trainLabels
)
trainClassWeightDict = {i: float(w) for i, w in enumerate(trainClassWeights)}
print(f"\nClass weights: {trainClassWeightDict}\n")"""

# -----------------------------
# 6) Model option A: small custom CNN
# -----------------------------
def make_custom_cnn(inputShape=(*IMAGE_SIZE, 3), numClasses=13):
    inputs = tf.keras.Input(shape=inputShape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(numClasses, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

# -----------------------------
# 6b) Model option B: Transfer learning (MobileNetV2)
# -----------------------------
def make_transfer_model(inputShape=(*IMAGE_SIZE, 3), numClasses=13):
    base = tf.keras.applications.MobileNetV2(
        input_shape=inputShape, include_top=False, weights='imagenet'
    )
    base.trainable = False   # freeze backbone initially
    inputs = tf.keras.Input(shape=inputShape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # mobilenet preprocess
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(numClasses, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------------
# 6c) Model option C: Custom Made Model
# -----------------------------
def makeCustomModel(inputShape = (*IMAGE_SIZE, 3), numClasses = 13):
    model = tf.keras.Sequential()
    #model = tf.keras.Model()
    print(f"inputShape:{inputShape}")
    model.add(tf.keras.layers.Input(shape = (50, 50, 3)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(13, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model
    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min",
                                            patience=5,
                                            restore_best_weights=True)


    history = model.fit(train_images_ds,
                        epochs=25,
                        callbacks=[earlystopping],
                        validation_split=.8)

    print("history:", history)

    #test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    #print('\nTest accuracy:', test_acc)

    model.save('./ChessModel2Save.keras')


# Choose model:
model = makeCustomModel()    # try transfer first; if images too few, try custom_cnn()
model.summary()

# -----------------------------
# 7) Compile, callbacks and train
# -----------------------------
"""
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])"""

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(SAVE_DIR / 'best_model.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = model.fit(
    trainDS,
    validation_data=validationDS,
    epochs=NUM_EPOCHS,
    class_weight=trainClassWeightDictionary,
    callbacks=callbacks
)

# -----------------------------
# 8) Evaluate & save mapping
# -----------------------------
model.evaluate(testDS)
model.save(SAVE_DIR / "final_model.keras")

print("Saved model and class_names.json in", SAVE_DIR)
