import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow.python.platform import gfile

inputImage = scipy.misc.imread('test.png')

oneSnapshot = inputImage[0:32, 0:32, :] / 255.0

classifierName= 'streetNumberClassifier'
imageWidth = inputImage.shape[0]
imageHeight = inputImage.shape[1]

with tf.Session() as session:   
  with gfile.FastGFile(("model/%s_prod.pb" % (classifierName)), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    session.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    output = session.graph.get_tensor_by_name("Output/softmax:0")
    prediction = session.run(output, {'tfTrainingData:0': oneSnapshot[None, :, :, :], 'keep_prob:0': 1.0})
    print prediction
    print np.argmax(prediction[0])
    '''
    for i in range(0, imageWidth-32, 10):
      for j in range(0, imageHeight-32, 10):
        partialImage = inputImage[i:i+32, j:j+32, :] / 255.0
        partialImage = partialImage[None, :, :, :]
        output = persisted_sess.graph.get_tensor_by_name("Output/softmax:0")
        prediction = persisted_sess.run(output, {'tfTrainingData:0': partialImage, 'keep_prob:0': 1.0})
        print inputImage[i:i+32, j:j+32, :]
        scipy.misc.imsave(('outfile%d-%d.jpg' % (i,j)), inputImage[i:i+32, j:j+32, :])
        print i, j, '=========================', np.argmax(prediction[0])
    '''
# ==========================================================================================================================
# End of code
# ==========================================================================================================================