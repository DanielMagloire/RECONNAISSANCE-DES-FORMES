# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:16:16 2017

@author: PAUL
"""
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from skimage import io

import cv2, os
import numpy as np
from PIL import Image
import glob
from os.path import splitext

# Utilisation du fichier Haar Cascade d'OpenCV pour la detection
cascadePath = "haarcascade_frontalface_default.xml"
faceDetectClassifier = cv2.CascadeClassifier(cascadePath)

# Methode de repartition d'images en train et test: 50% chacun
def get_img(root_path='att_faces', image_suffix='*.pgm'):
    train = [path for path in glob.glob(root_path + '/*/' + image_suffix) if int(os.path.basename(splitext(path)[0])) <= 5]
    test = [path for path in glob.glob(root_path + '/*/' + image_suffix) if int(os.path.basename(splitext(path)[0])) > 5]
        
    return (train, test)

# Methode de reconstruction de la base (Train et Test)    
def build_dataset(paths, new_path):
    #reconstruction de la base
    for img_path in paths:
        img = os.path.split(os.path.split(img_path)[1].split("/")[0])[1]
        nbr = int(os.path.split(os.path.split(img_path)[0].split("/")[0])[1].replace("s", ""))
        image = Image.open(img_path).convert('L')
        # Conversion de l'image au format numpy array
        image_np = np.array(image, 'uint8')
        cv2.imwrite(new_path+"/s."+str(nbr)+"."+img, image_np)
        
    image_paths = [os.path.join(new_path, path) for path in os.listdir(new_path)]
    
    return image_paths
        
# Methode de reconnaissance de visages
def get_img_and_labels(image_paths):
    faces = []
    labels = []
    for image_path in image_paths:
        # Lecture de l'image et conversion en niveau de gris
        face_img = io.imread(image_path, as_grey=True)
        label = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(face_img)
        labels.append(label)  
    return faces, np.array(labels)

    # methode de detection des visages
def get_final_data(faces):
    final_data = []
    for i in faces:
        facePoints = faceDetectClassifier.detectMultiScale(i)
        for (x, y, w, h) in facePoints:
            cropped = i[y:y + h, x:x + w]
        final_data.append(cropped)
    return final_data

# split de la base en apprentissage et test
trn, tst = get_img(root_path='att_faces', image_suffix='*.pgm')

# reconstruction de la base d'apprentissage dans un dossier train
print("** Construction de la base d'apprentissage **")

trn_img_paths = build_dataset(trn, "train")

print("** Construction de la base d'apprentissage terminee **")

# reconstruction de la base de test dans un dossier test
print("** Construction de la base de test **")

tst_img_paths = build_dataset(tst, "test")

print("** Construction de la base de test terminee **")

print("train = %i et test = %i"%(len(trn_img_paths), len(tst_img_paths)))

# recuperation des faces et identifiants
trn_faces, y_train = get_img_and_labels(trn_img_paths)
tst_faces, y_test = get_img_and_labels(tst_img_paths)

x_train = np.array(trn_faces)
x_test = np.array(tst_faces)

# parametres du cnn
batch_size = 128
num_classes = 41
epochs = 10

# dimensions des images en entree
img_rows, img_cols = 92, 112

# redimensionnement des donnees
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# conversion des vecteurs de classe en matrices binaires
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# construction des couches CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy = %f and loss = %f"%(accuracy, loss))
