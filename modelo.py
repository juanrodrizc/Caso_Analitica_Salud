# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 10:45:04 2023

@author: alejandrs
"""
import numpy as np
import joblib ### para cargar array


########Paquetes para NN #########
import tensorflow as tf #!pip install tensorflow
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo

### cargar bases_procesadas ####

x_train = joblib.load('x_train.pkl')
y_train = joblib.load('y_train.pkl')
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')

#Se verifica carga adecuada de los datos

x_train.shape
y_train.shape
x_test.shape
y_test.shape


"""Probar modelos de redes neuronales"""

##Normalizamos las variables 
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo

#Se utilizará una escala manual ya que el MinMaxScaler no permite las dimensiones de nuestro tipo de datos
 
x_train /=255 ### escalaro para que quede entre 0 y 1 con el valor máximo de la intensidad de un pixel en la bd
x_test /=255
np.product(x_train[1].shape)

##Definir arquitectura de la red neuronal e instanciar el modelo

y_train.shape
y_test.shape

modelo1=tf.keras.models.Sequential([ 
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]), #Se ingresan las dimensiones que tiene cada imágen, flatten permite "aplanar" las imágenes tridimensionales y así se convierten a unidimensionales
    tf.keras.layers.Dense(128, activation='relu'), #Se añade la capa a la red neuronal para empezar el proceso de aprendizaje
    tf.keras.layers.Dense(64, activation='relu'),#La activación se hace para que el aprendizaje no sea lineal 
    tf.keras.layers.Dense(1, activation='sigmoid') 
])

## Configura el optimizador y la función para optimizar 

modelo1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])

##Entrenar el modelo usando el optimizador y arquitectura definidas
modelo1.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))


#Evaluar el modelo
test_loss, test_acc, test_auc, test_recall, test_precision = modelo1.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)


#Matriz de confusión

pred_test=(modelo1.predict(x_test) > 0.50).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()
print(metrics.classification_report(y_test, pred_test))

##Seleccionar un indicador

### Se utilizará AUC: detección de positivos vs mala clasificaicón de negativos: porcentaje de los que tienen tumor que identifico vs los normales que dijo que tienen tumor

############Analisis problema ###########

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.8 ## porcentaje de neuronas que elimina

fc_model2=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#### configura el optimizador y la función para optimizar ##############


fc_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


#####Entrenar el modelo usando el optimizador y arquitectura definidas #########
fc_model2.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

####################### aplicar dos regularizaciones L2 y drop out

#Se puede aplicar regularización L1 o L2 a las capas densas para reducir el sobreajuste y mejorar el rendimiento del modelo en el conjunto de datos de validación.
###Penaliza el tamaño de los pesos, mientras más grande la penalización menores son los valores de los coeficientes

reg_strength = 0.0001

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.98 ## porcentaje de neuronas que utiliza 

fc_model3=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
##### configura el optimizador y la función para optimizar ##############
fc_model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

#####Entrenar el modelo usando el optimizador y arquitectura definidas #########
fc_model3.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))


