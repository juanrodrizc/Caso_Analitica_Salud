

import os  # para hacer lista de archivos en una ruta
import cv2  # para leer imagenes jpg y jpeg
from tqdm import tqdm  # mostrar barras de progreso en los bucles del for

def img2data(path, width=80):
    # define una función que recibe dos argumentos: "path" la ruta de las imágenes y "width" que es el tamaño de las imágenes
    
    rawImgs = [] # para almacenar el array que representa cada imágen
    labels = []  # para almacenar las etiquetas de cada imagen
    
    for label in os.listdir(path): # lista los elementos presentes en la ruta
        
        label_path = os.path.join(path, label)  # une la ruta principal "path" con el subdirectorio "label"
        
        for item in tqdm(os.listdir(label_path)): # recorre cada elemento en el subdirectorio "label_path" y muestra una barra de progreso
            
            file = os.path.join(label_path, item)   # crea la ruta completa de cada archivo en el subdirectorio "label_path"
            if not os.path.isfile(file) or os.path.splitext(file)[1].lower() not in ('.jpg', '.jpeg'):
                # verifica si el archivo es formato "jpg" o "jpeg", de lo contrario, pasa a la siguiente iteración
                continue
            
            img = cv2.imread(file)        # carga la imagen
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)    # convierte la imagen de BGR a RGB
            img = cv2.resize(img, (width, width))          # cambia el tamaño de la imagen 
            rawImgs.append(img)           # agrega la imagen procesada a la lista 
            labels.append([1 if label == 'tumor' else 0])   # agrega la etiqueta correspondiente a la lista "labels", donde 1 indica "tumor" y 0 indica "no tumor"
                    
    return rawImgs, labels  