# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 00:10:30 2017

@author: PAUL
"""
import cv2, os
import numpy as np
from PIL import Image
import glob
from os.path import splitext

# Utilisation du fichier Haar Cascade d'OpenCV pour la detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Pour la reconnaissance de visage, nous utilisons LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()

# Methode de repartition d'images en train et test: 50% chacun
def get_img(root_path='att_faces', image_suffix='*.pgm'):
    train = [path for path in glob.glob(root_path + '/*/' + image_suffix) if int(os.path.basename(splitext(path)[0])) <= 5]
    test = [path for path in glob.glob(root_path + '/*/' + image_suffix) if int(os.path.basename(splitext(path)[0])) > 5]
        
    return (train, test)

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
    print("** Apprentissage en cours **")
    for image_path in image_paths:
        # Lecture de l'image et conversion en niveau de gris
        face_img = Image.open(image_path).convert('L')
        # Conversion de l'image au format numpy array
        face_np = np.array(face_img, 'uint8')
        label = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(face_np)
        labels.append(label)
        cv2.imshow("Apprentissage ...", face_np)
        cv2.waitKey(10)
    
    print("** Apprentissage termine **")    
    return faces, np.array(labels)    


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
faces, labels = get_img_and_labels(trn_img_paths)

# phase d'apprentissage de la reconnaissance
recognizer.train(faces, labels)

# sauvegarde du model d'apprentissage
recognizer.save("recognizer/training_data.yml")

cv2.destroyAllWindows()

###########################################################################################
# Phase de test
###########################################################################################
reco = cv2.face.createLBPHFaceRecognizer()
# cahrgement du modele d'apprentissage
reco.load("recognizer\\training_data.yml")
correct = 0
incorrect = 0
for image_path in tst_img_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted = reco.predict(predict_image[y:y+h, x:x+w])
        #tx = 100.0 - conf
        tx = 0
        nbr_actual = int(os.path.split(image_path)[-1].split(".")[1])
        if nbr_actual == nbr_predicted:
            print ("s%i est correctement reconnu" % (nbr_actual))
            correct +=1
        else:
            print ("s%i est mal reconnu comme etant s%i" % (nbr_actual, nbr_predicted))
            incorrect +=1
        cv2.imshow("Visage reconnu", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)

print("Visages correctement reconnus: %i et mal reconnus: %i"%(correct, incorrect))