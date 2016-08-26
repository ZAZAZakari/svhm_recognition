import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.python.platform import gfile
import freeze_graph
import os
#trainMat = scipy.io.loadmat('processed_data/characterDetection/train.mat')
testMat = scipy.io.loadmat('processed_data/characterDetection/test.mat')

#trainingData = trainMat['data']
#trainingLabel = trainMat['label']
validateData = testMat['data']
validateLabel = testMat['label']

with tf.Session() as persisted_sess:
    print("load graph")
    with gfile.FastGFile("model/temp_out.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        output = persisted_sess.graph.get_tensor_by_name("Output/loss:0")
        prediction = persisted_sess.run(output, {'tfTrainingData:0': validateData[:1000, :, :, :], \
                                                 'tfTrainingLabels:0': validateLabel[:1000, :], \
                                                 'keep_prob:0': 1.0})
        print (prediction)
# ==========================================================================================================================
# End of code
# ==========================================================================================================================