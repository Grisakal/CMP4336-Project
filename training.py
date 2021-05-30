import os
import cv2
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import glob

# All paths
dataPath = "data/animalsCatDog"
trainingOutput = 'trainingOutput/trainingOutputAnimalsCatDog.h5'
trainingOutputLabels = 'trainingOutput/trainingOutputLabelsAnimalsCatDog.h5'

# Hu Moments for Shape Matching
def HUMoments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture for Gray Levels
def haralicTexture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# Color Histogram for Color distribution
def colorHistogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

trainingLabels = os.listdir(dataPath)
trainingLabels.sort()

featureVectors = []
labels = []

# Access each file and get all pictures
for fileName in trainingLabels:
    path = os.path.join(dataPath, fileName)

    onlyFiles = glob.glob(path + "/*.jpg") + glob.glob(path + "/*.jpeg") + glob.glob(path + "/*.png")
    print(fileName + " : " + str(len(onlyFiles)))

    for x in range(0, len(onlyFiles)):
        image = cv2.imread(onlyFiles[x])
        image = cv2.resize(image, tuple((500, 500)))

        if x % 200 == 0:
            print("Completed " + fileName + " : " + str(x) + "/" + str(len(onlyFiles)))

        # Add label name and feature vectors
        labels.append(fileName)
        featureVectors.append(np.hstack([colorHistogram(image), haralicTexture(image), HUMoments(image)]))

# Encoding Here
targetNames = np.unique(labels)
target = LabelEncoder().fit_transform(labels)

# Scale features in the range (0-1)
scaledInput = MinMaxScaler(feature_range=(0, 1)).fit_transform(featureVectors)

# Save training
outputFileData = h5py.File(trainingOutput, 'w')
outputFileData.create_dataset('dataset_1', data=np.array(scaledInput))

fileLabel = h5py.File(trainingOutputLabels, 'w')
fileLabel.create_dataset('dataset_1', data=np.array(target))

outputFileData.close()
fileLabel.close()

print("Training Complete!")
