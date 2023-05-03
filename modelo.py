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

### cargar bases_procesadas previamente####

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

#No hay sobreajuste(overfitting) ni subajuste(underfitting) por lo tanto no se realizará ninguna tecnica de regularización.

#Matriz de confusión

pred_test=(modelo1.predict(x_test) > 0.50).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'Normal'])
disp.plot()
print(metrics.classification_report(y_test, pred_test))

##Seleccionar un indicador

### Se utilizará AUC: detección de positivos vs mala clasificaicón de negativos: porcentaje de los que tienen tumor que identifico vs los normales que dijo que tienen tumor

############Analisis problema ###########

##########################################################
################ Redes convolucionales ###################
##########################################################

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Train the model for 10 epochs
cnn_model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))


#######probar una red con regulzarización


#####################################################
###### afinar hiperparameter ########################
#####################################################
####instalar paquete !pip install keras-tuner

import keras_tuner as kt


##### función con definicion de hiperparámetros a afinar

def build_model(hp):
    
    dropout_rate=hp.Float('DO', min_value=0.1, max_value= 0.4, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0005, step=0.0001)
    ####hp.Int
    ####hp.Choice
    

    model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
   
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
   
    model.compile(
        optimizer=opt, loss="binary_crossentropy", metrics=["AUC"],
    )
    return model




###########
hp = kt.HyperParameters()
build_model(hp)

tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=False, ## solo evalúe los hiperparámetros configurados
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld", 
)

tuner.search(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()



#################### Mejor redes ##############

joblib.dump(fc_best_model, 'fc_model.pkl')
joblib.dump(cnn_model,'cnn_model.pkl')

