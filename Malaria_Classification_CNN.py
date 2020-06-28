import os
import cv2
import wget
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt 
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile


wget.download("https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip")

zf = ZipFile('cell_images.zip', 'r')
zf.extractall()
zf.close()

parasitized_data = os.listdir('cell_images/Parasitized/')
print(parasitized_data[:10]) 

uninfected_data = os.listdir('cell_images/Uninfected/')
print('\n')
print(uninfected_data[:10])


print("Parasitized images:",len(parasitized_data))
print("Uninfected images:",len(uninfected_data))

plt.figure(figsize = (12,12))
for i in range(3):
    plt.subplot(1, 3, i+1)
    img = cv2.imread('cell_images/Parasitized' + "/" + parasitized_data[i])
    img = cv2.resize(img, (50, 50))
    plt.axis("off")
    plt.imshow(img)
    plt.title('PARASITIZED',color='red')
    
plt.show()

plt.figure(figsize = (12,12))
for i in range(3):
    plt.subplot(1, 3, i+1)
    img = cv2.imread('cell_images/Uninfected' + "/" + uninfected_data[i+1])
    img = cv2.resize(img, (50, 50))
    plt.axis("off")
    plt.imshow(img)
    plt.title('UNINFECTED', color='green')
   
plt.show()




data = []
labels = []
for img in parasitized_data:
    try:
        img_read = plt.imread('cell_images/Parasitized/' + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        data.append(img_array)
        labels.append(1)
    except:
        None
        
for img in uninfected_data:
    try:
        img_read = plt.imread('cell_images/Uninfected' + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        data.append(img_array)
        labels.append(0)
    except:
        None


image_data = np.array(data)
labels = np.array(labels)


#Shuffling data for better outcome while training :

index = np.arange(image_data.shape[0])
np.random.shuffle(index)
image_data = image_data[index]
labels = labels[index]

xtrain, xtest, ytrain, ytest = train_test_split(image_data, labels, test_size = 0.2, random_state = 100)

ytrain = np_utils.to_categorical(ytrain, num_classes = 2)
ytest = np_utils.to_categorical(ytest, num_classes = 2)

model = Sequential()

model.add(Conv2D(32, (3,3), activation = "relu", input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))


model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))

model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))

model.add(Conv2D(128, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.15))
model.add(Dense(2, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.summary()


#fit the model onto the dataset
h = model.fit(xtrain, ytrain, epochs = 2, batch_size = 32)

predictions = model.evaluate(xtest, ytest)

print(f'LOSS : {predictions[0]}')
print(f'ACCURACY : {predictions[1]}')




