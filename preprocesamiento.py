import numpy as np

import cv2 ### para leer imagenes jpg
from matplotlib import pyplot as plt ## para gráfciar imágnes
import funciones as fn ### funciones personalizadas, carga de imágenes
import joblib ### para descargar array
from PIL import Image

# Diseño de la solución
i=Image.open('diseno_solucion.png','r') 
i.show()

##### ver ejemplo de imágenes cargadas ######

img1 = cv2.imread('data_brain/testing/notumor/Te-no_0118.jpg')
img1_1 = cv2.imread('data_brain/testing/glioma/Te-gl_0010.jpg')
img1_2 = cv2.imread('data_brain/testing/pituitary/Te-pi_0010.jpg')
img1_3 = cv2.imread('data_brain/testing/meningioma/Te-me_0010.jpg')

img2 = cv2.imread('data_brain/training/notumor/Tr-no_0010.jpg')
img2_1 = cv2.imread('data_brain/training/glioma/Tr-gl_0010.jpg')
img2_2 = cv2.imread('data_brain/training/pituitary/Tr-pi_0010.jpg')
img2_3 = cv2.imread('data_brain/training/meningioma/Tr-me_0010.jpg')

##### ver ejemplo de imágenes cargadas ######

plt.imshow(img1)
plt.title('notumor')
plt.show()

plt.imshow(img1_1)
plt.title('glioma')
plt.show()

plt.imshow(img1_2)
plt.title('pituitary')
plt.show()

plt.imshow(img1_3)
plt.title('meningioma')
plt.show()

plt.imshow(img2)
plt.title('notumor')
plt.show()

plt.imshow(img2_1)
plt.title('glioma')
plt.show()

plt.imshow(img2_2)
plt.title('pituitary')
plt.show()

plt.imshow(img2_3)
plt.title('meningioma')
plt.show()

###### representación numérica de imágenes ####

img1.shape ### tamaño de imágenes
img1_2.shape 
img1_3.shape 

img2.shape
img2_2.shape #512,512,3
img2_3.shape

img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel

img2.max() ### máximo valor de intensidad en un pixel
img2.min() ### mínimo valor de intensidad en un pixel

np.prod(img1.shape) ### 151008 millones de observaciones cada imágen
np.prod(img2.shape) ### 151875 millones de observaciones cada imágen

#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes

img1 = cv2.resize(img1 ,(80,80))
plt.imshow(img1)
plt.title('notumor')
plt.show()

################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################

width = 80 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'data_brain/training/'
testpath = 'data_brain/testing/'

x_train, y_train= fn.img2data(trainpath) #Run in train
x_test, y_test = fn.img2data(testpath) #Run in test

#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train.shape
y_train.shape

x_test.shape
y_test.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "x_train.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(x_test, "x_test.pkl")
joblib.dump(y_test, "y_test.pkl")


