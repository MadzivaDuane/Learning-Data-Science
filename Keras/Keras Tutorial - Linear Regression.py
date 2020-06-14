#Keras Module
#Using Neural Networks for Linear Regression - Comparing accuracy to regular sklearn linear regression

"""Build A Model to Predict Fuel Efficiency"""
#import necessary packages 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

#you may run into issues installing tensorflow docs
"""Do the following:
1. If you get the following error:
    xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun 
run the following command in your terminal:
    xcode-select --install
and install the software for xcode

2. Then run the following command line in your terminal or IDE:
    pip install -q git+https://github.com/tensorflow/docs
then import the packages below"""

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

#get data
path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
cars_data = raw_dataset.copy()
cars_data.tail()

#check for missing values
cars_data.isnull().sum()
#for now lets drop all missing values
cars_data = cars_data.dropna()
#convert origin variable to categorical variable
cars_data["Origin"] = cars_data["Origin"].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
#create dummy variables
cars_data = pd.get_dummies(cars_data, prefix ='', prefix_sep='')

#split the data into training and testing data
train_data = cars_data.sample(frac = 0.8, random_state = 0)
test_data = cars_data.drop(train_data.index)  #drop essentially removes all the data in the trainig dataset and leaves the remaining indexes in the test dataset

#create a pairsplot to better understand the variables in the dataset 
sns.pairplot(train_data[["MPG", "Cylinders", "Displacement", "Weight"]])
sns.pairplot(train_data[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")   #smoother histograms

#gather basic statistics on each parameter
train_stats = train_data.describe()
train_stats.pop('MPG')   #looks specifically at stats for MPG - pop removes a specific index from the dataset or file 
#transpose for easier viewing
train_stats = train_stats.transpose()
#check for cylinder categories
train_data.Cylinders.unique()  #3,4,5,6,8 cylinder vehicles

#split features from targets
train_target = train_data.pop('MPG')
test_target = test_data.pop('MPG')

#given the range of the features in train_stats, it is a good idea to normalize the data
#to ensure that scales used are not important 
#create a function to normalize the data
def norm(x):
    return (x - train_stats['mean'])/ train_stats['std']  #expect the MPG column to be NaN becuase train_stats does not have MPG

normalized_train_data = norm(train_data)
normalized_test_data = norm(test_data)

#build the model
#Sequential - single input single output 
"""
 model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
1. the Dense functionality adds a layer, and in this case, the first dense argument is the input layer. You have to 
stipulate the shape of the incoming data bu using input_shape
2. The output layer needs specification of the dimensions of the output. In this case, we require a 1 dimensional output,
that is, a predicted value for fuel efficiency - MPG
"""

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),  #stipulate shape of input
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  #specify dimensions of output
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  """Every time a neural network finishes passing a batch through the network and generating prediction results, 
  it must decide how to use the difference between the results it got and the values it knows to be true to adjust 
  the weights on the nodes so that the network steps towards a solution. The algorithm that determines that step is known as the optimization algorithm."""   

  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
  return model

model = build_model()
#get a summary of the model
model.summary()  #there are a total of 4929 trainable parameters
#try the model on a batch of 10 samples
sample_batch = normalized_train_data[:10]
sample_result = model.predict(sample_batch)
sample_result
#the model is working, so we can train it on the training data
EPOCHS = 1000  #An epoch is a measure of the number of times all of the training vectors are used once to update the weights

training_model = model.fit(
  normalized_train_data, train_target,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

#we can visualize the training process 
history = pd.DataFrame(training_model.history)
history['epoch'] = training_model.epoch
history.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': training_model}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

plotter.plot({'Basic': training_model}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]') #from the plot, the validation error starts degrading after about 100 epochs, so we have to adjust our model to stop much earlier

#early stopping 
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

training_model_final = model.fit(normalized_train_data, train_target, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
#visualize the training process
plotter.plot({'Early Stopping': training_model_final}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')  #much better

#making final predictions using final model
predictions = model.predict(normalized_test_data).flatten()   #flatten() function is used to get a copy of an given array collapsed into one dimension.

a = plt.axes(aspect='equal')
plt.scatter(test_target, predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
#score the model using r2_score: R^2 regression coefficient
keras_score = r2_score(test_target, predictions)
r2_score(pd.DataFrame(test_target), pd.DataFrame(predictions.tolist(), columns = ["MPG"]))

#Using Statsmodel SMF Linear Regression tool - comparative method
#get data - preprocess the same way as above method
path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
cars_data = raw_dataset.copy()
#for now lets drop all missing values
cars_data = cars_data.dropna()
#convert origin variable to categorical variable
cars_data["Origin"] = cars_data["Origin"].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
#create dummy variables
cars_data = pd.get_dummies(cars_data, prefix ='', prefix_sep='')
#rename Model Year column
cars_data = cars_data.rename(columns={"Model Year": "Model_Year"})

#split the data into training and testing data
train_data = cars_data.sample(frac = 0.8, random_state = 0)
test_data = cars_data.drop(train_data.index)

#create model
lm_model_smf = smf.ols(formula = 'MPG ~ Cylinders + Displacement + Horsepower + Weight + Acceleration + Model_Year + Europe + USA + Japan', data = train_data).fit()
lm_model_smf.summary()

#make predictions
predictions_smf = lm_model_smf.predict(test_data.drop(columns = ["MPG"]))
#score Statsmodel SMF model
smf_score = r2_score(test_data.MPG, predictions_smf)

#final results
scores = pd.DataFrame({"Keras Linear Regression":[keras_score], "SMF Linear Regression": [smf_score]}).T
scores.columns = ["R^2 Score"]; scores

