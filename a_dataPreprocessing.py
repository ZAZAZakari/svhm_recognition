# ==========================================================================================================================
# a_dataPreprocessing.py
# Preprocessing the input raw data by normalizing and reshaping dataset to make it ready to train.
# It was then splited into two sets: training and validate for cross validation. Finally, the label
# matrix is one-hot encoded and all the dataset is saved to external pickle files
# ==========================================================================================================================
# [INPUT]  (2 files) train_32x32.mat, test_32x32.mat
# [OUTPUT] (6 files) trainingData.pickle, validateData.pickle, testingData.pickle,
#                    trainingLabel.pickle, validateLabel.pickle, testingLabel.pickle
# ==========================================================================================================================
import scipy.io
import toPickle
import numpy as np
from sklearn.cross_validation import train_test_split

# ==========================================================================================================================
# [FUNCTION] Reshape Dataset from (32, 32, 3, m) to (m, 32, 32, 3) where m is the sample size 
# ==========================================================================================================================
# [INPUT]  numUniqueLabels			    [Integer] How many value exists in the label list? Example: 4 for the list below
#		   labelList               		[1D Array] Example: [0,1,2,3,2,1]
# [OUTPUT] labelListEncoded				[2D Array] Example: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]
#===========================================================================================================================
def reshapeDataset(dataset):
	rChannel = np.transpose(dataset[:,:,0,:], (2,0,1))		# Getting the red channel from the big matrix 
	gChannel = np.transpose(dataset[:,:,1,:], (2,0,1))  	# Getting the green channel from the big matrix
	bChannel = np.transpose(dataset[:,:,2,:], (2,0,1))		# Getting the blue channel from the big matrix
	return np.concatenate((rChannel[:,:,:,None], gChannel[:,:,:,None], bChannel[:,:,:,None]), axis=3)

# ==========================================================================================================================
# [FUNCTION] One-hot encoding a list of integers
#            Reference: https://en.wikipedia.org/wiki/One-hot
# ==========================================================================================================================
# [INPUT]  numUniqueLabels			    [Integer] How many value exists in the label list? Example: 4 for the list below
#		   labelList               		[1D Array] Example: [0,1,2,3,2,1]
# [OUTPUT] labelListEncoded				[2D Array] Example: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]
#===========================================================================================================================
def oneHotEncoding(numUniqueLabels, labelList):
    # [Step 0] Initialize the 2D Array with zeros #
    labelListEncoded = np.zeros((len(labelList), numUniqueLabels))
    # [Step 1] Loop through the label list, mark the corresponding position with 1.0
    for dataIndex, labelIndex in enumerate(labelList):	
        labelListEncoded[dataIndex][labelIndex] = 1.0
    return np.array(labelListEncoded, dtype=np.float32)
	
# ==========================================================================================================================
# ==========================================================================================================================
# ALGORITHM STARTS HERE
# ==========================================================================================================================
# [Step 1] Load raw dataset from the external .mat file 
# ==========================================================================================================================
trainingMat = scipy.io.loadmat('train_32x32.mat')
testingMat = scipy.io.loadmat('test_32x32.mat')

# ==========================================================================================================================
# [Step 2] Normalize the input training dataset by dividing each element by 255.0
# ==========================================================================================================================
trainingData = trainingMat['X'] / 255.0 
trainingLabel = trainingMat['y']
testingData = testingMat['X'] / 255.0
testingLabel = testingMat['y']
del trainingMat			# release resource
del testingMat			# release resource

# ==========================================================================================================================
# [Step 3] Reshaping the dataset from (32, 32, 3, m) to (m, 32, 32, 3), to make it easier to split and train
# ==========================================================================================================================
trainingData = reshapeDataset(trainingData)
testingData = reshapeDataset(testingData)

# ==========================================================================================================================
# [Step 4] Divide the whole training into two sets, 70% for training, 30% for validate, stratified to make sure 
#          the number of data for each class is divided equally
# ==========================================================================================================================
trainingData, validateData, trainingLabel, validateLabel = train_test_split(trainingData, trainingLabel, test_size=0.3, stratify=trainingLabel)

# ==========================================================================================================================
# [Step 5] One-hot encoding each label
# ==========================================================================================================================
trainingLabel = oneHotEncoding(10, trainingLabel.flatten()-1)
validateLabel = oneHotEncoding(10, validateLabel.flatten()-1)
testingLabel = oneHotEncoding(10, testingLabel.flatten()-1)

# ==========================================================================================================================
# [Step 6] Save all the data to external pickle files
# ==========================================================================================================================
toPickle.save(trainingData, 'trainingData', 'processed_data')
toPickle.save(trainingLabel, 'trainingLabel', 'processed_data')
toPickle.save(validateData, 'validateData', 'processed_data')
toPickle.save(validateLabel, 'validateLabel', 'processed_data')
toPickle.save(testingData, 'testingData', 'processed_data')
toPickle.save(testingLabel, 'testingLabel', 'processed_data')

# ==========================================================================================================================
# End of code
# ==========================================================================================================================