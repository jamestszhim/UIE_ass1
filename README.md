# UIE_ass1
Preprocessing: mask the source image to get segmented images and depth image, apply gaussian filter to remove noise
Feature Extraction: use HOG and daisy provided by skimage.feature
Dimension Reduction: use PCA to reduce dimension to 450 features
Parameter tuning: Use grid search to obtain the best parameters
Predict: Use the tuned model to predict test data

# data
This source code does not include the training data or test data. Add training data or test data under a new directry "data" and specifiy the
file name in SVM.py
