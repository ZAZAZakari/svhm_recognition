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
import maybe_download
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
# [Step 0] Check if the needed dataset exists, if not, download from cloud
# ==========================================================================================================================
maybe_download.download('processed_data', 'https://storage.googleapis.com/yobi3d-deep-learning/raw/test_32x32.mat')
maybe_download.download('processed_data', 'https://storage.googleapis.com/yobi3d-deep-learning/raw/train_32x32.mat')
maybe_download.download('processed_data', 'https://storage.googleapis.com/yobi3d-deep-learning/raw/negative_samples.mat')

# ==========================================================================================================================
# [Step 1] Load raw dataset from the external .mat file 
# ==========================================================================================================================
trainingMat = scipy.io.loadmat('processed_data/train_32x32.mat')
testingMat = scipy.io.loadmat('processed_data/test_32x32.mat')
negativeSamples_data = scipy.io.loadmat('processed_data/negative_samples.mat')['data']
negativeSamples_label = np.array([[11]] * negativeSamples_data.shape[0])

# ==========================================================================================================================
# [Step 2] Reshaping the dataset from (32, 32, 3, m) to (m, 32, 32, 3), to make it easier to split and train
# ==========================================================================================================================
trainingMat['X'] = reshapeDataset(trainingMat['X'])
testingMat['X'] = reshapeDataset(testingMat['X'])
trainingMat['X'] = np.concatenate((trainingMat['X'], testingMat['X'], negativeSamples_data), axis=0)
trainingMat['y'] = np.concatenate((trainingMat['y'], testingMat['y'], negativeSamples_label), axis=0)
del testingMat
del negativeSamples_data
del negativeSamples_label

# ==========================================================================================================================
# [Step 3] Normalize the input training dataset by dividing each element by 255.0
# ==========================================================================================================================
trainingMat['X'] = trainingMat['X'] / 255.0
trainingMat['y'] = trainingMat['y'].flatten()-1

# ==========================================================================================================================
# [Step 4] Divide the whole training into two sets, 70% for training, 30% for validate, stratified to make sure 
#          the number of data for each class is divided equally
# ==========================================================================================================================
trainingData, validateData, trainingLabel, validateLabel = train_test_split(trainingMat['X'], trainingMat['y'], test_size=0.3, stratify=trainingMat['y'])
del trainingMat
validateData, testingData, validateLabel, testingLabel = train_test_split(validateData, validateLabel, test_size=0.5, stratify=validateLabel)

# ==========================================================================================================================
# [Step 5] Save the processed splited matrix to external .mat files
# ==========================================================================================================================
scipy.io.savemat('processed_data/training.mat', {'data': trainingData, 'label': oneHotEncoding(11, trainingLabel)})
del trainingData, trainingLabel

scipy.io.savemat('processed_data/validate.mat', {'data': validateData, 'label': oneHotEncoding(11, validateLabel)})
del validateData, validateLabel

scipy.io.savemat('processed_data/test.mat', {'data': testingData, 'label': oneHotEncoding(11, testingLabel)})
del testingData, testingLabel
# ==========================================================================================================================
# End of code
# ==========================================================================================================================