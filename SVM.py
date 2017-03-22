# -*- coding: utf-8 -*-

#!/usr/bin/env python

import numpy as np
#import NoiseModel as noise
import Features as features
import SVM_Model as model
import DataSaver as saver
import pylab as pl
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage.color import rgb2gray

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

pl.close('all')
# data_generator = data.DataGenerator()
data_saver = saver.DataSaver('data', 'a1_dataTrain.pkl')
training_data = data_saver.restore_from_file()
training_labels = training_data['gestureLabels'][0:1000]
n_samples = training_labels.shape[0]
n_image_size = training_data['depth'][0].shape         

a = np.zeros((3, 3))
a[1, 1] = 1
 
# Preprocessing: extract the masked image for training input
training_input = np.zeros([n_samples, 2, n_image_size[0], n_image_size[1]])
for i in range(0, n_samples):
    segmentedUser = training_data['segmentation'][i]
    mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
    mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
    mask3 = mask3.transpose((1,2,0))
    # Masked depth
    training_input[i][0] = rgb2gray(gaussian(training_data['rgb'][i]*mask3, sigma=2, multichannel=True))
    training_input[i][1] = gaussian(training_data['depth'][i]*mask2, sigma=2, multichannel=False)

# Feature Extraction
print('Extracting Features')
svm = model.SVM_Model()
svm.set_feature_vector([features.HOG(), features.Daisy()])
feature_matrix = svm.compute_feature_matrix(training_input)

# Split data into training and validation
X_train, X_test, labels_train, labels_test = train_test_split(feature_matrix, training_labels, test_size=0.2, random_state=1)
print(str(X_train.shape) + " - " + str(X_test.shape))

# Dimension Reduction
print('Reducing Dimension')
pca = PCA(n_components = 450).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Tune the parameters
# Optimize the parameters by cross-validation
print('=========SVM===========')
parameters = [
    {'kernel': ['linear'], 'C': [100]}
]
kparameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 1, 10, 100]},
    {'kernel': ['linear'], 'C': [0.01, 1, 10, 100]}
]

# Grid search object with SVM classifier.
clf = GridSearchCV(SVC(), parameters, cv=10)
clf.fit(X_train, labels_train)

print("Best parameters set found on training set:")
print(clf.best_params_)
print()

means_valid = clf.cv_results_['mean_test_score']
stds_valid = clf.cv_results_['std_test_score']
means_train = clf.cv_results_['mean_train_score']

print("Grid scores:")
for mean_valid, std_valid, mean_train, params in zip(means_valid, stds_valid, means_train, clf.cv_results_['params']):
    print("Validation: %0.3f (+/-%0.03f), Training: %0.3f  for %r" % (mean_valid, std_valid, mean_train, params))
print()

labels_test, labels_predicted = labels_test, clf.predict(X_test)
print("Test Accuracy [%0.3f]" % ((labels_predicted == labels_test).mean()))

print("runing test")
data_saver = saver.DataSaver('data', 'a1_dataTest.pkl')
testing_data = data_saver.restore_from_file()
n_samples = testing_data['subjectLabels'].shape[0]

testing_input = np.zeros([n_samples, 2, n_image_size[0], n_image_size[1]])
for i in range(0, n_samples):
    segmentedUser = testing_data['segmentation'][i]
    mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
    mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
    mask3 = mask3.transpose((1,2,0))
    # Masked depth
    testing_input[i][0] = rgb2gray(gaussian(testing_data['rgb'][i]*mask3, sigma=2, multichannel=True))
    testing_input[i][1] = gaussian(testing_data['depth'][i]*mask2, sigma=2, multichannel=False)

# Feature Extraction
print('Extracting Features')
feature_matrix = svm.compute_feature_matrix(testing_input)

print("PCA")
real_test_data_transformed = pca.transform(feature_matrix)

print("Predict")
real_predicted_labels = clf.predict(real_test_data_transformed)

print("Write to CSV")
with open('output.csv','w') as file:
    file.write('Id,Prediction')
    for i in range(0, n_samples):
       file.write('\n')
       file.write(str(i+1))
       file.write(',')
       file.write(str(real_predicted_labels[i]))

pl.show()
