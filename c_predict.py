import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow.python.platform import gfile
from PIL import Image, ImageFont, ImageDraw


inputImage = Image.open('2376.png')
inputImageMatrix = np.array(inputImage)

draw = ImageDraw.Draw(inputImage)
font = ImageFont.truetype("Silom.ttf", 30)

classifierName= 'streetNumberClassifier'
imageWidth = inputImageMatrix.shape[0]
imageHeight = inputImageMatrix.shape[1]

with tf.Session() as session:   
  with gfile.FastGFile(("model/%s_prod.pb" % (classifierName)), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    session.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    '''
    output = session.graph.get_tensor_by_name("Output/softmax:0")
    prediction = session.run(output, {'tfTrainingData:0': oneSnapshot[None, :, :, :], 'keep_prob:0': 1.0})
    print prediction
    print np.argmax(prediction[0])
    '''
    
    for i in range(0, imageWidth-32, 32):
      for j in range(0, imageHeight-32, 20):
        partialImage = inputImageMatrix[i:i+32, j:j+32, :] / 255.0
        partialImage = partialImage[None, :, :, :]
        output = session.graph.get_tensor_by_name("Output/softmax:0")
        prediction = session.run(output, {'tfTrainingData:0': partialImage, 'keep_prob:0': 1.0})
        draw.rectangle((j, i, j+20, i+32), fill='black')
        draw.text((j, i), str(np.argmax(prediction)+1),(255,255,255), font=font)
        print i, j, '===================', np.argmax(prediction)
        scipy.misc.imsave(('outfile%d-%d.png' % (i,j)), inputImageMatrix[i:i+32, j:j+32, :])
    inputImage.save('sample-out.png')
# ==========================================================================================================================
# End of code
# ==========================================================================================================================