
# For Machine learning 2023
# %%
from math import sqrt
from numpy import concatenate
import tensorflow as tf
import math
import os
                                                              
import numpy as np
from tensorflow import keras
from keras import backend as K
import random
#from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# %%
# X is your input and y is your output
X = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
y = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

COST_FUNCTION = "mean_squared_error"
    #design Network      
#   https://keras.io/api/layers/core_layers/dense/
# You must have 2 hidden units.  You fill in where ? exists with your choice.
# Use a small learning rate less than 0.2
model = keras.Sequential()

model.add(keras.Input(shape=(4,))) # Modern way for input
model.add(keras.layers.Dense(2, activation='sigmoid')) # Suggest tanh or sigmoid
# Output layer.  Your activations need to be close to 0 or 1
model.add(keras.layers.Dense(4, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=COST_FUNCTION, optimizer = opt) 
print(model.summary())
# %%
history = model.fit(X, y, validation_data=(X,y),epochs=200, batch_size=1,
                    verbose=1, shuffle=False)

model.save('autoenc'+'.h5')
# %%
predictions = model.predict(X)
predictions=np.round(predictions)
print("Predictions are\n",predictions)
print("Model Weights")
print (model.get_weights())
# %%
# This works in jupyter notebook only
get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
layer_output = get_1st_layer_output(X)[0]
print("Hidden output: ", layer_output)
new_input=layer_output

# %%
# Build model to predict from this compressed input (of 2 values) with a single or more layers
# neural network
neural_net = keras.Sequential()
neural_net.add(keras.Input(shape=(2,)))
neural_net.add(keras.layers.Dense(32, activation='sigmoid'))
neural_net.add(keras.layers.Dense(4, activation='sigmoid'))
neural_net.compile(loss=COST_FUNCTION, optimizer=keras.optimizers.Adam(learning_rate=0.01))
print(neural_net.summary())
# %%
history_nn = neural_net.fit(new_input, y, validation_data=(new_input,y), epochs=100, batch_size=16,
                         verbose=1, shuffle=False)
# %%
predictions_nn = neural_net.predict(new_input)
predictions_nn = np.round(predictions_nn)
print("Predictions are\n",predictions_nn)
print("Model Weights")
print (neural_net.get_weights())
# %%
'''
Use a scikit learn decision tree and the hidden unit activations of your
autoencoder to predict the desired 4 outputs.  Info on the DecisionTree classifier is at:
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier

crdt = DecisionTreeClassifier(criterion='entropy',random_state=0)
'''
crdt = DecisionTreeClassifier(criterion='entropy',random_state=0)
history_dt = crdt.fit(new_input, y)
# %%
predictions_dt = crdt.predict(new_input)
predictions_dt = np.round(predictions_dt)
print("Predictions are\n", predictions_dt)
# %%
#plot_tree(crdt)

# %%
