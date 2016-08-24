import pickle
import tensorflow as tf

trainingData = pickle.load(open('processed_data/trainingData'))
validateData = pickle.load(open('processed_data/validateData'))
trainingLabel = pickle.load(open('processed_data/trainingLabel'))
validateLabel = pickle.load(open('processed_data/validateLabel'))

graph = tf.Graph()
with graph.as_default():
    # Declaring placeholders 
    tfTrainingData = tf.placeholder(tf.float32, [None, 32, 32, 3])
    tfTrainingLabels = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
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
    output = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tfTrainingLabels))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9).minimize(loss)
    correctPrediction = tf.equal(tf.argmax(output, 1), tf.argmax(tfTrainingLabels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
	
batchSize = 128
with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    step = 0
    
    while (True):
        offset = (step * batchSize) % (trainy.shape[0] - batchSize)
        # slicing the whole training set into batches # 
        batchData = trainX[offset:(offset + batchSize), :, :, :]  
        batchLabel = trainy[offset:(offset + batchSize), :]
        
        _, stepLoss, stepAcc = session.run([optimizer, loss, accuracy], 
                                    feed_dict={tfTrainingData: batchData,
                                               tfTrainingLabels: batchLabel,
                                               keep_prob: 1.0})
        step += 1
        if (step == 1 or step % 100 == 0):
            validLoss, validAcc = session.run([loss, accuracy], 
                                    feed_dict={tfTrainingData: validateData,
                                               tfTrainingLabels: validateLabel,
                                               keep_prob: 1.0})
            print ("Step %5d: Loss = %10.6f, Train Acc. = %10.6f, Valid Acc. = %10.6f" % (step, stepLoss, stepAcc, validAcc))
            
        if (step > 10000):
            save_path = saver.save(session, "abc.ckpt")
            break
# ==========================================================================================================================
# End of code
# ==========================================================================================================================