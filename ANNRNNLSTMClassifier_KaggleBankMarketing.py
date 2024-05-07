#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:38:54 2021

@author: nitinsinghal
"""
# RNN ANN vs RF for classification - Kaggle Portuguese bank marketing data
#### RNN ###

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import layers, models, losses, metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# Load the data
train_data = pd.read_csv('./Portugese Bank Data - TRAIN.csv')
test_data = pd.read_csv('./Portugese Bank Data - TEST.csv')

# Perform data wrangling - remove duplicate values and clean null values
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)

train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Split categorical data for one hot encoding
train_data_cat = train_data.select_dtypes(exclude=['int64','float64'])
train_data_num = train_data.select_dtypes(include=['int64','float64'])

test_data_cat = test_data.select_dtypes(exclude=['int64','float64'])
test_data_num = test_data.select_dtypes(include=['int64','float64'])

# Encode the categorical features using OneHotEncoder. Use the same encoder for train and test set
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(train_data_cat)
train_data_cat = pd.DataFrame(ohe.transform(train_data_cat))
test_data_cat = pd.DataFrame(ohe.transform(test_data_cat))

# Merge encoded categorical data with mueric data
train_data_ohe = train_data_num.join(train_data_cat)
test_data_ohe = test_data_num.join(test_data_cat)

# Setup the traing and test X, y datasets
X_train = train_data_ohe.iloc[:,:-1].values
y_train = train_data_ohe.iloc[:,-1].values
X_test = test_data_ohe.iloc[:,:-1].values
y_test = test_data_ohe.iloc[:,-1].values

# Scale all the data as some features have larger range compared to the rest
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
batch_size = 32
units = 100
output_size = 1

rnnmodel = models.Sequential()

rnnmodel.add(layers.LSTM(units, input_shape=(input_dim,1), return_sequences=True))
rnnmodel.add(layers.BatchNormalization())
rnnmodel.add(layers.LSTM(units, return_sequences=True))
rnnmodel.add(layers.BatchNormalization())
rnnmodel.add(layers.LSTM(units))
rnnmodel.add(layers.BatchNormalization())
#rnnmodel.add(layers.Dropout(rate=0.2))

rnnmodel.add(layers.Dense(100, activation='relu'))
rnnmodel.add(layers.BatchNormalization())
#rnnmodel.add(layers.Dropout(rate=0.2))
rnnmodel.add(layers.Dense(output_size, activation='sigmoid'))

rnnmodel.summary()

rnnmodel.compile(loss=losses.BinaryCrossentropy(),
              optimizer="Adam",
              metrics=[metrics.BinaryAccuracy()])

history = rnnmodel.fit(X_train, y_train, batch_size=batch_size, epochs=2, validation_data=(X_test, y_test))

print(history.params)
print(history.history.keys())

# Plot the train/test accuracy to see marginal improvement
plt.plot(history.history['binary_accuracy'], label='binary_accuracy')
plt.plot(history.history['val_binary_accuracy'], label = 'val_binary_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('RNN Train Vs Test')
plt.show()

# Evaluate the rnnmodel using the test set
# test_loss, test_acc = rnnmodel.evaluate(X_test, y_test, verbose=1)
# print('Evaluate Acc: ', test_acc)

y_pred = rnnmodel.predict(X_test, verbose=1)
print('LSTM Pred Prob: ', y_pred)

####### ANN ##########

annmodel = models.Sequential()
annmodel.add(layers.Dense(50, activation='relu'))
annmodel.add(layers.Dense(50, activation='relu'))
annmodel.add(layers.Dense(50, activation='relu'))
annmodel.add(layers.Dense(output_size, activation='sigmoid'))

annmodel.compile(loss=losses.BinaryCrossentropy(from_logits=False),
              optimizer="Adam",
              metrics=[metrics.BinaryAccuracy()])

history = annmodel.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_data=(X_test, y_test))

print(history.params)
print(history.history.keys())

# Plot the train/test accuracy to see marginal improvement
plt.plot(history.history['binary_accuracy'], label='binary_accuracy')
plt.plot(history.history['val_binary_accuracy'], label = 'val_binary_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('ANN Train Vs Test')
plt.show()

y_pred_prob = annmodel.predict(X_test, verbose=1)
y_pred = np.where(y_pred_prob<0.5, 0, 1)
print('ANN accuracy_score: ', accuracy_score(y_test, y_pred))
print('ANN classification_report: \n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('ANN Confusion matrix: \n', cm)

auc = roc_auc_score(y_test, y_pred_prob)
print('ANN AUC: ', auc)

###### Using Random Forest Classifier #######
# Create the random forest classifier object and fit the training data
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

# Make predictions using the input X test features
y_pred = classifier.predict(X_test)

# Accuracy score
print('Random Forest accuracy_score: ', accuracy_score(y_test, y_pred))
print('Random Forest classification_report: \n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('RF Confusion matrix: \n', cm)

auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
print('RF AUC: ', auc)



