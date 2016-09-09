import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow.python.platform import gfile

#validateData = pickle.load(open('processed_data/validateData.pickle'))
#validateLabel = pickle.load(open('processed_data/validateLabel.pickle'))
inputImage = scipy.misc.imread('2376.png')
print (inputImage.shape)

oneSnapshot = inputImage[0:32, 0:32, :]

classifierName= 'streetNumberClassifier'

imageWidth = inputImage.shape[0]
imageHeight = inputImage.shape[1]

with tf.Session() as persisted_sess:   
  with gfile.FastGFile(("model/%s_prod.pb" % (classifierName)), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    for i in range(0, imageWidth-32, 10):
      for j in range(0, imageHeight-32, 10):
        
        output = persisted_sess.graph.get_tensor_by_name("Output/softmax:0")
        prediction = persisted_sess.run(output, {'tfTrainingData:0': inputImage[None, i:i+32, j:j+32, :], \
                                                 'keep_prob:0': 1.0})
        
        if (np.argmax(prediction) > 1):
          print i, j, '=========================', np.argmax(prediction), np.max(prediction)
# ==========================================================================================================================
# End of code
# ==========================================================================================================================