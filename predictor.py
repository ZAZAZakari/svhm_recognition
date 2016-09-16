import numpy as np
import tensorflow as tf
import scipy.misc
import maybe_download
import os
from tensorflow.python.platform import gfile
from PIL import Image, ImageFont, ImageDraw

'''
This is a tensorflow predictor, object which import a pre-trained tensorflow graph and 
execute prediction with a given image name
'''
class Predictor():
	'''
	Intialize the predictor, it includes steps to import the pre-trained network and get the tensorflow session ready 
	'''
	def __init__(self, classifierName):
		# [Step 1] Download the pre-trained .pb file 
		self.classifierName = classifierName
		maybe_download.download('model', 'https://storage.googleapis.com/yobi3d-deep-learning/models/%s_prod.pb' % (self.classifierName))
		
		# [Step 2] Read the .pb file and import it as the graph_def
		graphFileReader = gfile.FastGFile(("model/%s_prod.pb" % (self.classifierName)), "rb")
		self.graph_def = tf.GraphDef()
		self.graph_def.ParseFromString(graphFileReader.read())
		
		# [Step 3] Initialize a tensorflow session, with the graph imported 
		self.session = tf.Session()
		self.session.graph.as_default()
		tf.import_graph_def(self.graph_def, name='')
	
	'''
	Make prediction with the given fileName
	[INPUT] the image file to predict: 'uploads/$imgFileName'
	[ACTION] make prediction and overlay the predicted result on the input image, save the result to another image
	[OUTPUT] the image file after predict: 'predictions/$imgFileName' 
	'''
	def predict(self, imageFileName):
		# [Step 1] Read the input image and convert the image to a 3D matrix, keep a copy of PIL object for drawing
		inputImage = Image.open('%s/uploads/%s' % (os.getcwd(), imageFileName))
		inputImageMatrix = np.array(inputImage)
		
		# [Step 2] Read the drawing object
		draw = ImageDraw.Draw(inputImage)
		font = ImageFont.truetype("Silom.ttf", 30)
		
		# [Step 3] Slide the image with a 32x32 window, predict each partial image and see if it is a number 
		imageWidth, imageHeight, imageChannel = inputImageMatrix.shape
		for i in range(0, imageWidth-32, 32):
			for j in range(0, imageHeight-32, 20):
				# [Step 3(a)] Get the partial image while the window is sliding
				partialImage = inputImageMatrix[i:i+32, j:j+32, :] / 255.0
				partialImage = partialImage[None, :, :, :]
				
				# [Step 3(b)] Make prediction on the partial image
				output = self.session.graph.get_tensor_by_name("Output/softmax:0")
				prediction = self.session.run(output, {'tfTrainingData:0': partialImage, 'keep_prob:0': 1.0})
				
				# [Step 3(c)] Draw a text with background rectangle to visualize the predicitons
				draw.rectangle((j, i, j+20, i+32), fill='black')
				draw.text((j, i), str(np.argmax(prediction)+1), fill='white', font=font)
		
		# [Step 4] Save the predictions image
		if not os.path.exists(os.getcwd() + '/predictions'):
			os.makedirs(os.getcwd() + '/predictions')
		inputImage.save('%s/predictions/%s' % (os.getcwd(), imageFileName))
# ==========================================================================================================================
# End of code
# ==========================================================================================================================