import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Settings
path = 'Data'
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)
batchSizeVal = 50
epochsVal = 10
#

# list of images
images = []
# list of class id of img - [1, 2, 3, ...]
classNum = []

dataList = os.listdir(path)
noOfClassNum = len(dataList)

for i in range(1, noOfClassNum + 1):
    sublist = os.listdir(path + '/' + str(i))
    for j in sublist:
        currentImg = cv2.imread(path + '/' + str(i) + '/' + j)
        currentImg = cv2.resize(currentImg, imageDimensions[:2])
        images.append(currentImg)
        classNum.append(i)
    print("Importing: ", i)
print("Imported ", len(images), "images")

images = np.array(images)
classesNum = np.array(classNum)

print(images.shape)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(images, classesNum, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

numOfSamples = []
for i in range(1, noOfClassNum + 1):
    numOfSamples.append(len(np.where(y_train == i)[0]))
print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClassNum), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


x_train = np.array(list(map(pre_processing, x_train)))
x_test = np.array(list(map(pre_processing, x_test)))
x_validation = np.array(list(map(pre_processing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

# adding augment
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)

dataGenerator.fit(x_train)

y_train = to_categorical(y_train - 1, noOfClassNum)
y_test = to_categorical(y_test - 1, noOfClassNum)
y_validation = to_categorical(y_validation - 1, noOfClassNum)


# using Linear regression model
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClassNum, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
model.summary()


history = model.fit(x_train,
                    y_train,
                    batch_size=batchSizeVal,
                    epochs=epochsVal,
                    validation_data=(x_validation, y_validation),
                    shuffle=1)

#
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

# save model
model.save('OCRmodel.h5')
