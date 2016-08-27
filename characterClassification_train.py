import numpy as np
import tensorflow as tf
import scipy.io
import freeze_graph
import os

trainMat = scipy.io.loadmat('processed_data/characterRecognition/train.mat')
validateMat = scipy.io.loadmat('processed_data/characterRecognition/validate.mat')

trainingData = trainMat['data']
validateData = validateMat['data']
trainingLabel = trainMat['label']
validateLabel = validateMat['label']

del trainMat
del validateMat
graph = tf.Graph()
with graph.as_default():
    # Declaring placeholders 
    tfTrainingData = tf.placeholder(tf.float32, [None, 32, 32, 3], name='tfTrainingData')
    tfTrainingLabels = tf.placeholder(tf.float32, [None, 10], name='tfTrainingLabels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
    def bias_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
    def conv2d(x, W, s):
        return tf.nn.conv2d(x, W, strides=s, padding='SAME')
    
    def max_pool_2x2(x, k):
        return tf.nn.max_pool(x, ksize=k, strides=[1,2,2,1], padding='SAME')
    
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])
    
    h_conv1 = tf.nn.relu(conv2d(tfTrainingData, W_conv1, [1,1,1,1]) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1, [1,2,2,1])
    
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1,1,1,1]) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2, [1,2,2,1])

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, [1,1,1,1]) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3, [1,2,2,1])
    
    h_pool2_flat = tf.reshape(h_pool3, [-1, 2048])
    W_fc1 = weight_variable([2048, 2048])
    b_fc1 = bias_variable([2048])
    
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([2048, 1024])
    b_fc2 = bias_variable([1024])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)
    h_fc_2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    W_fc3 = weight_variable([1024, 10])
    b_fc3 = bias_variable([10])  
    logits = (tf.matmul(h_fc_2_drop, W_fc3) + b_fc3)
    with tf.name_scope('Output'):
        output = tf.nn.softmax(logits, name='softmax')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tfTrainingLabels), name='loss')
        correctPrediction = tf.equal(tf.argmax(output, 1), tf.argmax(tfTrainingLabels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32), name='accuracy')
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

batchSize = 128
with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    step = 0
    while (True):
        offset = (step * batchSize) % (trainingLabel.shape[0] - batchSize)
        # slicing the whole training set into batches # 
        batchData = trainingData[offset:(offset + batchSize),:,:,:]  
        batchLabel = trainingLabel[offset:(offset + batchSize),:]
        
        _, stepLoss, stepAcc = session.run([optimizer, loss, accuracy], \
                            feed_dict={tfTrainingData: batchData, tfTrainingLabels: batchLabel, keep_prob: 0.8})
        step += 1
        if (step == 1 or step % 100 == 0):
            validLoss, validAcc = session.run([loss, accuracy], \
                                    feed_dict={tfTrainingData: validateData[:1000,:,:,:], \
                                               tfTrainingLabels: validateLabel[:1000,:] , keep_prob: 1.0})
            print ("Step %5d: Loss = %10.6f, TrainAcc = %10.6f, validLoss = %10.6f, validAcc = %10.6f" % \
                    (step, stepLoss, stepAcc, validLoss, validAcc))
            
            if (step > 90000):
                save_path = saver.save(session, "model/characterClassification_temp.ckpt", global_step=0)
                tf.train.write_graph(session.graph_def, 'model', 'characterClassification_temp.pb', False)
                freeze_graph.freeze_graph('model/characterClassification_temp.pb', '', True, \
                                      'model/characterClassification_temp.ckpt-0', 'Output/softmax,Output/loss,Output/accuracy', \
                                      'save/restore_all', 'save/Const:0', "model/characterClassification_prod.pb", False, "") 
                os.system('rm model/characterClassification_temp.*')
                #save_path = saver.save(session, "bcd.ckpt")
                break
# ==========================================================================================================================
# End of code
# ==========================================================================================================================