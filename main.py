import os
import tkinter

import cv2
import h5py
import numpy as np
import mahotas
import warnings
import threading
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#GUI Lib
from tkinter import *
import time
from tkinter import filedialog as fd

# Ignore warnings for clean console output (helps with debugging)
warnings.filterwarnings('ignore')

# Main window root widget
root = Tk(className = 'ProjectX')
root.geometry('650x380')
root['background'] = '#064420'

# Checkbox Variables
var1 = tkinter.IntVar()
var2 = tkinter.IntVar()
var3 = tkinter.IntVar()
var4 = tkinter.IntVar()
var5 = tkinter.IntVar()
var6 = tkinter.IntVar()
var7 = tkinter.IntVar()

# File paths
dataPath = "data/animalsCatDog"
trainingOutputDir = 'trainingOutput/trainingOutputAnimalsCatDog.h5'
trainingOutputLabelsDir = 'trainingOutput/trainingOutputLabelsAnimalsCatDog.h5'

# Get data from default path
trainingLabels = os.listdir(dataPath)
trainingLabels.sort()

featureVectors = []
labels = []

# Import training files
trainingOutput = h5py.File(trainingOutputDir, 'r')
trainingOutputLabels = h5py.File(trainingOutputLabelsDir, 'r')

globalFeatures = np.array(trainingOutput['dataset_1'])
globalLabels = np.array(trainingOutputLabels['dataset_1'])

trainingOutput.close()
trainingOutputLabels.close()

# Splitting data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(globalFeatures),
                                                                                          np.array(globalLabels),
                                                                                          test_size=0.10,
                                                                                          random_state=10)

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

# Select a file
def selectFile():
    global path
    path = fd.askopenfilename()
    if path:
        fileLabel.configure( text = "File Selected: " + path, font = 10)
        startButton.configure(state = NORMAL)

# Tale model performances and compare in a graphic
def compareModels():
    # Create all the machine learning models
    allModels = []
    allModels.append(('Gaussian NB', GaussianNB()))
    allModels.append(('Logistic Regression', LogisticRegression(random_state=10)))
    allModels.append(('SVC', SVC(random_state=10)))
    allModels.append(('Random Forest Classifier', RandomForestClassifier(n_estimators=100, random_state=10)))
    allModels.append(('K Neighbours Classifier', KNeighborsClassifier()))
    allModels.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=10)))
    allModels.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))

    statusText.configure(text="Started training.", font=10)
    print("Started training.")

    # Compare models
    count = 1
    for name, model in allModels:
        statusText.configure(text="Now trying : " + name + " - " + str(count) + "/7", font=10)
        print("Now trying : " + name)

        kfold = KFold(n_splits=10, random_state=10, shuffle=True)
        results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="accuracy")

        featureVectors.append(results)
        labels.append(name)
        count = count + 1
    statusText.configure(text="Comparison Completed!", font=10)
    #statusText.insert(END, "Comparison Completed!")

# Print the comparison graphic
def showComparison():
    # Print comparison figure
    fig = pyplot.figure()
    fig.suptitle('Model Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(featureVectors)
    ax.set_xticklabels(labels)
    pyplot.show()

# Run the file through each model chosen by the user
def startTest():
    # Append chosen models
    allModels = []
    if var1.get():
        allModels.append(('Gaussian NB', GaussianNB()))
    if var2.get():
        allModels.append(('Logistic Regression', LogisticRegression(random_state=10)))
    if var3.get():
        allModels.append(('SVC', SVC(random_state=10)))
    if var4.get():
        allModels.append(('Random Forest Classifier', RandomForestClassifier(n_estimators=100, random_state=10)))
    if var5.get():
        allModels.append(('K Neighbours Classifier', KNeighborsClassifier()))
    if var6.get():
        allModels.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=10)))
    if var7.get():
        allModels.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))

    count = 1
    for name, model in allModels:
        statusText.configure(text="Calculating using : " + name + " - " + str(count) + "/" + str(len(allModels)), font=10)
        model.fit(trainDataGlobal, trainLabelsGlobal)

        image = cv2.imread(path)
        image = cv2.resize(image, tuple((512, 512)))

        features = np.hstack([colorHistogram(image), haralicTexture(image), HUMoments(image)])
        features = np.reshape(features, (2, -1))

        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledFeature = scaler.fit_transform(features)

        result = model.predict(rescaledFeature.reshape(1,-1))[0]

        # Print image
        cv2.rectangle(image, (0, 0), (350, 25), (0, 0, 0), -1)
        cv2.putText(image, name, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv2.rectangle(image, (0, 25), (60, 50), (0, 0, 0), -1)
        cv2.putText(image, trainingLabels[result], (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pyplot.show()
        count = count + 1

    statusText.configure(text="Process Completed! Please select a new picture and/or model.", font=10)

t1 = threading.Thread(target=compareModels,daemon=True)
def startThread():
    t1.start()

# Labels
mainLabel = Label(root, text="Project X", font=48, bg='#064420', fg = '#fdfaf6')
mainLabel.place(relx=0.5, rely=0.1, anchor=CENTER)

fileLabel = Label(root, text="Select a file and model", font=10, bg='#064420', fg = '#fdfaf6')
fileLabel.place(relx=0.5, rely=0.2, anchor=CENTER)

statusText = Label(root, text="Status: Ready", font=10, bg='#064420', fg = '#fdfaf6')
statusText.place(relx=0.5, rely=0.3, anchor=CENTER)

# Buttons
selectButton = Button(root, text="Select File", font=24, bg='#fdfaf6', fg = '#064420', width=16, command = selectFile)
selectButton.place(relx=0.35, rely=0.5, anchor=CENTER)

selectButton = Button(root, text="Compare Models", font=24, bg='#fdfaf6', fg = '#064420', width=16, command = startThread)
selectButton.place(relx=0.65, rely=0.5, anchor=CENTER)

showGraphics = Button(root, text="Show Graphic", font=24, bg='#fdfaf6', fg = '#064420', width=16, state = NORMAL, command = showComparison)
showGraphics.place(relx=0.35, rely=0.6, anchor=CENTER)

startButton = Button(root, text="Start", font=24, bg='#fdfaf6', fg = '#064420', width=16, state = DISABLED, command = startTest)
startButton.place(relx=0.65, rely=0.6, anchor=CENTER)

# Checkboxes
c1 = Checkbutton(root, text='Gaussian NB',variable=var1, onvalue=1, offvalue=0)
c1.place(relx=0.25, rely=0.7, anchor=CENTER)
c2 = Checkbutton(root, text='Logistic Regression',variable=var2, onvalue=1, offvalue=0)
c2.place(relx=0.5, rely=0.7, anchor=CENTER)
c3 = Checkbutton(root, text='SVC',variable=var3, onvalue=1, offvalue=0)
c3.place(relx=0.75, rely=0.7, anchor=CENTER)
c4 = Checkbutton(root, text='Random Forest Classifier',variable=var4, onvalue=1, offvalue=0)
c4.place(relx=0.25, rely=0.8, anchor=CENTER)
c5 = Checkbutton(root, text='K Neighbours Classifier',variable=var5, onvalue=1, offvalue=0)
c5.place(relx=0.5, rely=0.8, anchor=CENTER)
c6 = Checkbutton(root, text='Decision Tree Classifier',variable=var6, onvalue=1, offvalue=0)
c6.place(relx=0.75, rely=0.8, anchor=CENTER)
c7 = Checkbutton(root, text='Linear Discriminant Analysis',variable=var7, onvalue=1, offvalue=0)
c7.place(relx=0.5, rely=0.9, anchor=CENTER)

root.mainloop()


