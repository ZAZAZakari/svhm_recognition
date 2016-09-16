import numpy as np
import tensorflow as tf
import scipy.misc
import maybe_download
import os
from tensorflow.python.platform import gfile
from PIL import Image, ImageFont, ImageDraw

class Predictor():
	def __init__(self, classifierName):
		self.classifierName = classifierName
		maybe_download.download('model', 'https://storage.googleapis.com/yobi3d-deep-learning/models/%s_prod.pb' % (self.classifierName))
		graphFileReader = gfile.FastGFile(("model/%s_prod.pb" % (self.classifierName)), "rb")
		self.graph_def = tf.GraphDef()
		self.graph_def.ParseFromString(graphFileReader.read())
		
		self.session = tf.Session()
		self.session.graph.as_default()
		tf.import_graph_def(self.graph_def, name='')
		
	def predict(self, imageFileName):
		inputImage = Image.open('%s/uploads/%s' % (os.getcwd(), imageFileName))
		inputImageMatrix = np.array(inputImage)
		
		draw = ImageDraw.Draw(inputImage)
		font = ImageFont.truetype("Silom.ttf", 30)
		
		imageWidth = inputImageMatrix.shape[0]
		imageHeight = inputImageMatrix.shape[1]
		
		for i in range(0, imageWidth-32, 32):
			for j in range(0, imageHeight-32, 20):
				partialImage = inputImageMatrix[i:i+32, j:j+32, :] / 255.0
				partialImage = partialImage[None, :, :, :]
				output = self.session.graph.get_tensor_by_name("Output/softmax:0")
				prediction = self.session.run(output, {'tfTrainingData:0': partialImage, 'keep_prob:0': 1.0})
				draw.rectangle((j, i, j+20, i+32), fill='black')
				draw.text((j, i), str(np.argmax(prediction)+1), fill='white', font=font)
				print i, j, '===================', np.argmax(prediction)
		if not os.path.exists(os.getcwd() + '/predictions'):
			os.makedirs(os.getcwd() + '/predictions')
		
		inputImage.save('%s/predictions/%s' % (os.getcwd(), imageFileName))
# ==========================================================================================================================
# End of code
# ==========================================================================================================================