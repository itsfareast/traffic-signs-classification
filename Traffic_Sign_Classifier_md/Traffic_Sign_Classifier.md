

```python
import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.utils import shuffle
from scipy.misc import imread, imsave, imresize
from skimage import exposure
import warnings 
```


```python
def preprocess(X):
    '''
    - convert images to grayscale, 
    - scale from [0, 255] to [0, 1] range, 
    - use localized histogram equalization as images differ 
      in brightness and contrast significantly
    ADAPTED FROM: http://navoshta.com/traffic-signs-classification/
    '''

    #Convert to grayscale, e.g. single channel Y
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]

    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)
    
    #adjust histogram
    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i]) 
            
    return X

def reshape(x): # Add a single grayscale channel
  return x.reshape(x.shape + (1,))
```


```python
######################################
##   LOAD AND PREPROCESS DATA       #
####################################

#''' 
#The pickled data is a dictionary with 4 key/value pairs
#
#features: the images pixel values, (width, height, channels)
#labels: the label of the traffic sign
#sizes: the original width and height of the image, (width, height)
#coords: coordinates of a bounding box around the sign in the image, 
#        (x1, y1, x2, y2). 
#        Based the original image (not the resized version).
#'''
```


```python
######################
#!!   EDIT ME!   !! #
####################

class_name_file = './signnames.csv'
training_file = "data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"

training_preprocessed_file = "data/X_train_preprocessed.p"
validation_preprocessed_file = "data/X_valid_preprocessed.p"
testing_preprocessed_file = "data/X_test_preprocessed.p" 
```


```python
# LOAD DATA SETS TO MEMORY

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```


```python
# Preprocess the data to improve feature extraction... This might take a while..

X_preprocessed = preprocess(X_train)
X_train_preprocessed = reshape(X_preprocessed)
print("training set preprocessing complete!", X_train_preprocessed.shape)

X_valid_preprocessed = preprocess(X_valid)
X_valid_preprocessed = reshape(X_valid_preprocessed)
print("cross validation set preprocessing complete!", X_valid_preprocessed.shape)

X_test_preprocessed = preprocess(X_test)
X_test_preprocessed = reshape(X_test_preprocessed)
print("test set preprocessing complete!", X_test_preprocessed.shape)
```

    training set preprocessing complete! (34799, 32, 32, 1)
    cross validation set preprocessing complete! (4410, 32, 32, 1)
    test set preprocessing complete! (12630, 32, 32, 1)



```python
# Save the preprocessed data set, so we don't have to preprocess it everytime 

pickle.dump(X_train_preprocessed, open(training_preprocessed_file, "wb" ))
pickle.dump(X_valid_preprocessed, open(validation_preprocessed_file, "wb" ))
pickle.dump(X_test_preprocessed, open(testing_preprocessed_file, "wb" ))
```


```python
# If the preprocessed data exists, we can just open them up
# no need to preprocess them everytime  

with open(training_preprocessed_file, mode='rb') as f:
    X_train_preprocessed = pickle.load(f)
with open(validation_preprocessed_file, mode='rb') as f:
    X_valid_preprocessed = pickle.load(f)
with open(testing_preprocessed_file, mode='rb') as f:
    X_test_preprocessed = pickle.load(f)
```


```python
######################################
#          DATA EXPLORATION         #
####################################
```


```python
# We can see some basic statistics about the data sets here 

n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_classes = len(np.unique(y_test))

number_of_images, image_width, image_height, number_of_color_channels = X_train.shape
image_shape = image_width, image_height, number_of_color_channels

print()
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_valid :", X_valid.shape)
print("y_valid :", y_valid.shape)
print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

print()
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    
    X_train : (34799, 32, 32, 3)
    y_train : (34799,)
    X_valid : (4410, 32, 32, 3)
    y_valid : (4410,)
    X_test  : (12630, 32, 32, 3)
    y_test  : (12630,)
    
    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43



```python
# Let's count how many samples per classification the training set has
# We can also map the label value representation to actual human label from CSV file
# in essence: y: 10 is the STOP sign
# Notice that the data set is not shuffled, images with the same class are adjacent to each
# other in the list. So a running count is useful to to know what range of position 
# an images of the same class occupies. 

classes_lineup = []
count_per_class = [0]*n_classes
running_counts = [0]*n_classes
class_names = []

for i in range(0, len(y_train)):
    if y_train[i] not in classes_lineup:
        classes_lineup.append(y_train[i])
    count_per_class[y_train[i]]+=1

count_before = 0
for n in classes_lineup:
    running_counts[n] = count_per_class[n] + count_before
    count_before = running_counts[n]
    
with open(class_name_file) as _f:
    rows = csv.reader(_f, delimiter=',')
    next(rows, None)
    for i, row in enumerate(rows):
        assert(i==int(row[0]))
        class_names.append(row[1]) 
```


```python
def show_images(X, end, total, images_per_row = 30, images_per_col = 15,
                H = 20, W = 1, its_gray = False):    
    number_of_images = images_per_row * images_per_col
    figure, axis = plt.subplots(images_per_col, images_per_row, figsize=(H, W))
    figure.subplots_adjust(hspace = .2, wspace=.001)
    axis = axis.ravel()
    
    for i in range(number_of_images):
        index = random.randint(end - total, end)
        image = X[index]
        axis[i].axis('off')
        if its_gray:
          axis[i].imshow(image.reshape(32,32), cmap='gray')
        else:
          axis[i].imshow(image)
```


```python
def plot_histogram(data, name):
  class_list = range(n_classes)
  label_list = data.tolist()
  counts = [label_list.count(i) for i in class_list]
  plt.bar(class_list, counts)
  plt.xlabel(name)
  plt.show()
```


```python
# More useful statistics about the data set
# value representation of that class/label 
# number of images with that class/label
# running count, and name of the class/label

print("-----------------------------------------------------------")
print("|%-*s | RUNNING | #   | NAME'" % (6, 'COUNT'))
print("-----------------------------------------------------------")

for n in classes_lineup:
    print("|%-*s | %-*s | %-*s | %s " % (6, count_per_class[n], 7, running_counts[n], 3, n, class_names[n]))
```

    -----------------------------------------------------------
    |COUNT  | RUNNING | #   | NAME'
    -----------------------------------------------------------
    |210    | 210     | 41  | End of no passing 
    |690    | 900     | 31  | Wild animals crossing 
    |330    | 1230    | 36  | Go straight or right 
    |540    | 1770    | 26  | Traffic signals 
    |450    | 2220    | 23  | Slippery road 
    |1980   | 4200    | 1   | Speed limit (30km/h) 
    |300    | 4500    | 40  | Roundabout mandatory 
    |330    | 4830    | 22  | Bumpy road 
    |180    | 5010    | 37  | Go straight or left 
    |360    | 5370    | 16  | Vehicles over 3.5 metric tons prohibited 
    |1260   | 6630    | 3   | Speed limit (60km/h) 
    |180    | 6810    | 19  | Dangerous curve to the left 
    |1770   | 8580    | 4   | Speed limit (70km/h) 
    |1170   | 9750    | 11  | Right-of-way at the next intersection 
    |210    | 9960    | 42  | End of no passing by vehicles over 3.5 metric tons 
    |180    | 10140   | 0   | Speed limit (20km/h) 
    |210    | 10350   | 32  | End of all speed and passing limits 
    |210    | 10560   | 27  | Pedestrians 
    |240    | 10800   | 29  | Bicycles crossing 
    |240    | 11040   | 24  | Road narrows on the right 
    |1320   | 12360   | 9   | No passing 
    |1650   | 14010   | 5   | Speed limit (80km/h) 
    |1860   | 15870   | 38  | Keep right 
    |1260   | 17130   | 8   | Speed limit (120km/h) 
    |1800   | 18930   | 10  | No passing for vehicles over 3.5 metric tons 
    |1080   | 20010   | 35  | Ahead only 
    |360    | 20370   | 34  | Turn left ahead 
    |1080   | 21450   | 18  | General caution 
    |360    | 21810   | 6   | End of speed limit (80km/h) 
    |1920   | 23730   | 13  | Yield 
    |1290   | 25020   | 7   | Speed limit (100km/h) 
    |390    | 25410   | 30  | Beware of ice/snow 
    |270    | 25680   | 39  | Keep left 
    |270    | 25950   | 21  | Double curve 
    |300    | 26250   | 20  | Dangerous curve to the right 
    |599    | 26849   | 33  | Turn right ahead 
    |480    | 27329   | 28  | Children crossing 
    |1890   | 29219   | 12  | Priority road 
    |690    | 29909   | 14  | Stop 
    |540    | 30449   | 15  | No vehicles 
    |990    | 31439   | 17  | No entry 
    |2010   | 33449   | 2   | Speed limit (50km/h) 
    |1350   | 34799   | 25  | Road work 



```python
#PLOT 350 RANDOM IMAGES from training set
show_images(X_train, len(X_train), len(X_train), 
            images_per_row = 30, images_per_col = 15, 
            H = 20, W = 10)
```


![png](output_14_0.png)



```python
#PLOT 350 RANDOM IMAGES from PREPROCESSED training set
i = np.copy(X_train_preprocessed)

show_images(i, len(i), len(i), images_per_row = 30, images_per_col = 15, 
            H = 20, W = 10, its_gray = True)
```


![png](output_15_0.png)



```python
#PLOT 10 RANDOM IMAGES each per classification
for n in classes_lineup:
    show_images(X_train, running_counts[n], count_per_class[n], 
                images_per_row = 10, images_per_col = 1, H = 20, W = 20)
```

    /home/carnd/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/matplotlib/pyplot.py:524: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)



![png](output_16_1.png)



![png](output_16_2.png)



![png](output_16_3.png)



![png](output_16_4.png)



![png](output_16_5.png)



![png](output_16_6.png)



![png](output_16_7.png)



![png](output_16_8.png)



![png](output_16_9.png)



![png](output_16_10.png)



![png](output_16_11.png)



![png](output_16_12.png)



![png](output_16_13.png)



![png](output_16_14.png)



![png](output_16_15.png)



![png](output_16_16.png)



![png](output_16_17.png)



![png](output_16_18.png)



![png](output_16_19.png)



![png](output_16_20.png)



![png](output_16_21.png)



![png](output_16_22.png)



![png](output_16_23.png)



![png](output_16_24.png)



![png](output_16_25.png)



![png](output_16_26.png)



![png](output_16_27.png)



![png](output_16_28.png)



![png](output_16_29.png)



![png](output_16_30.png)



![png](output_16_31.png)



![png](output_16_32.png)



![png](output_16_33.png)



![png](output_16_34.png)



![png](output_16_35.png)



![png](output_16_36.png)



![png](output_16_37.png)



![png](output_16_38.png)



![png](output_16_39.png)



![png](output_16_40.png)



![png](output_16_41.png)



![png](output_16_42.png)



![png](output_16_43.png)



```python
#PLOT HISTOGRAM OF EACH DATA SET
plot_histogram(y_train, name = "TRAINING SET: number of data points per class")
plot_histogram(y_valid, name = "CROSS VALIDATION SET: number of data points per class")
plot_histogram(y_test, name = "TEST SET: number of data points per class")
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



```python
######################################
#      NETWORK ARCHITECTURE         #
####################################
```


```python
def convolution(x, W, b, s = 1, with_relu = True, with_maxpool = False):
    result = tf.nn.conv2d(x, W, strides = [1, s, s, 1], padding = 'SAME')
    result = tf.nn.bias_add( result, b)
    if with_relu:
        result = tf.nn.relu(result)
    if with_maxpool:
        result = tf.nn.max_pool( result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return result

def fully_connected(x, W, b, keep_prob, do_flatten = True, with_relu = True):
    if do_flatten:
        x = flatten(x)
    result = tf.add(tf.matmul(x, W), b) 
    if with_relu:
        result = tf.nn.relu(result)
    result = tf.nn.dropout(result, keep_prob)
    return result
    
def network(x, W, b, dropout_prob):
    r = convolution(x, W['wc1'], b['bc1'], with_maxpool = True)
    r = convolution(r, W['wc2'], b['bc2'], with_maxpool = True)
    r = fully_connected(r, W['wf1'], b['bf1'], keep_prob = dropout_prob)
    r = fully_connected(r, W['wf2'], b['bf2'], keep_prob = dropout_prob, do_flatten = False, with_relu = False)
    return r 
```


```python
output_size = 43 #number of classifiers/labels - n_classes
c = 1         
fs1 = 5       
fs2 = 5        
depth1 = 64  #32
depth2 = 32  #64
fc_out = 256 #1024


weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(fs1, fs1, c, depth1), mean = 0, stddev = 0.1)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(fs2, fs2, depth1, depth2), mean = 0, stddev = 0.1)),
    'wf1': tf.Variable(tf.truncated_normal(shape=(8*8*depth2, fc_out), mean = 0, stddev = 0.1)),
    'wf2': tf.Variable(tf.truncated_normal(shape=(fc_out, output_size), mean = 0, stddev = 0.1))  
}

biases = {
    'bc1': tf.Variable(tf.zeros(depth1)),
    'bc2': tf.Variable(tf.zeros(depth2)),
    'bf1': tf.Variable(tf.zeros(fc_out)),
    'bf2': tf.Variable(tf.zeros(output_size))
}

#CONV1_INPUT: 32x32x1 OUTPUT:32x32xdepth1 MAXPOOLOUTPUT: 16x16xdepth1
#CONV2_INPUT: 16x16xdepth1 OUTPUT: 16x16xdepth2 MAXPOOLOUTPUT: 8x8xdepth2
#FC1_INPUT: 8x8xdepth2 OUTPUT: 8x8xdepth2
#FC1_INPUT: 8x8xdepth2 OUTPUT: n_classes
```


```python
################################################
#      NETWORK TRAINING     AND TESTING       #
##############################################
```


```python
LEARNING_RATE = 0.00005

EPOCHS = 180     
BATCH_SIZE = 256 #512

IMAGE_SIZE = 32
NUMBER_OF_CLASSES = 43           #n_classes
NUMBER_OF_TRAINING_DATA = 34799  #len(y_train)

LR = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, 1))
y = tf.placeholder(tf.int32, shape = (None))
one_hot_y = tf.one_hot(y, NUMBER_OF_CLASSES)
keep_prob = tf.placeholder(tf.float32) 

saver = tf.train.Saver()

logits = network(x, weights, biases, dropout_prob = keep_prob)

CROSS_ENTROPY_OPERATION = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
LOSS_OPERATION = tf.reduce_mean(CROSS_ENTROPY_OPERATION)
OPTIMIZER_OPERATION = tf.train.AdamOptimizer(learning_rate = LR)
TRAINING_OPERATION = OPTIMIZER_OPERATION.minimize(LOSS_OPERATION)
INFERENCE_OPERATION = tf.argmax(logits, 1)
CORRECT_PREDICTION_OPERATION = tf.equal(INFERENCE_OPERATION, tf.argmax(one_hot_y, 1))
ACCURACY_OPERATION = tf.reduce_mean(tf.cast(CORRECT_PREDICTION_OPERATION, tf.float32))
```


```python
def get_batch(X_data, y_data, start, BATCH_SIZE):
    end = start + BATCH_SIZE
    return X_data[start:end], y_data[start:end]

def evaluate(X_data, y_data):
    
    total_accuracy = 0
    total_samples = len(X_data)
    sess = tf.get_default_session()
    
    for start in range(0, total_samples, BATCH_SIZE):        
        batch_x, batch_y = get_batch(X_data, y_data, start, BATCH_SIZE) 
        params = {x: batch_x, y: batch_y, keep_prob: 1.0}
        accuracy = sess.run(ACCURACY_OPERATION, feed_dict= params)
        total_accuracy += (accuracy * len(batch_x))
    
    return total_cost/ total_samples, total_accuracy / total_samples
```


```python
# TRAIN THE MODEL

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):
        
        X_data, y_data = shuffle(X_train_preprocessed, y_train)
        
        for start in range(0, NUMBER_OF_TRAINING_DATA, BATCH_SIZE):
            
            batch_x, batch_y = get_batch(X_data, y_data, start, BATCH_SIZE)
            params = {x: batch_x, y: batch_y, keep_prob: 0.75, LR: LEARNING_RATE}
            _, loss = sess.run([TRAINING_OPERATION, LOSS_OPRATION], feed_dict = params)
            
        validation_accuracy, validation_cost = evaluate(X_valid_preprocessed, y_valid)
        
        print("{:3d}".format(epoch), "VA = {:.3f}".format(validation_accuracy), 
              "last training cost = {:.3f}".format(loss))
        
    saver.save(sess, './model')
    print("Model saved")
```

      0 VA = 0.159 cost = 3.509865
      1 VA = 0.345 cost = 2.983721
      2 VA = 0.466 cost = 2.573606
      3 VA = 0.539 cost = 2.439859
      4 VA = 0.611 cost = 2.019744
      5 VA = 0.669 cost = 1.819499
      6 VA = 0.709 cost = 1.808188
      7 VA = 0.745 cost = 1.705296
      8 VA = 0.753 cost = 1.440065
      9 VA = 0.780 cost = 1.349036
     10 VA = 0.797 cost = 1.452468
     11 VA = 0.811 cost = 1.323506
     12 VA = 0.817 cost = 1.190047
     13 VA = 0.828 cost = 1.278049
     14 VA = 0.839 cost = 1.063583
     15 VA = 0.846 cost = 1.317498
     16 VA = 0.855 cost = 1.058808
     17 VA = 0.861 cost = 1.183124
     18 VA = 0.865 cost = 0.948065
     19 VA = 0.868 cost = 1.077286
     20 VA = 0.875 cost = 1.035328
     21 VA = 0.879 cost = 1.009082
     22 VA = 0.882 cost = 0.990742
     23 VA = 0.885 cost = 0.922477
     24 VA = 0.889 cost = 1.035396
     25 VA = 0.893 cost = 0.976457
     26 VA = 0.896 cost = 0.839368
     27 VA = 0.895 cost = 0.979501
     28 VA = 0.900 cost = 0.790099
     29 VA = 0.904 cost = 0.728587
     30 VA = 0.905 cost = 0.786695
     31 VA = 0.905 cost = 0.770512
     32 VA = 0.908 cost = 1.001017
     33 VA = 0.909 cost = 0.700880
     34 VA = 0.915 cost = 0.777972
     35 VA = 0.910 cost = 0.719677
     36 VA = 0.916 cost = 0.732804
     37 VA = 0.919 cost = 0.778068
     38 VA = 0.916 cost = 0.843048
     39 VA = 0.921 cost = 0.794358
     40 VA = 0.923 cost = 0.807919
     41 VA = 0.922 cost = 0.738739
     42 VA = 0.922 cost = 0.717630
     43 VA = 0.924 cost = 0.833165
     44 VA = 0.924 cost = 0.664568
     45 VA = 0.924 cost = 0.751857
     46 VA = 0.929 cost = 0.840010
     47 VA = 0.932 cost = 0.673881
     48 VA = 0.930 cost = 0.758305
     49 VA = 0.929 cost = 0.839326
     50 VA = 0.936 cost = 0.882707
     51 VA = 0.931 cost = 0.705218
     52 VA = 0.933 cost = 0.696979
     53 VA = 0.932 cost = 0.687764
     54 VA = 0.931 cost = 0.624728
     55 VA = 0.934 cost = 0.772527
     56 VA = 0.934 cost = 0.735324
     57 VA = 0.936 cost = 0.710019
     58 VA = 0.933 cost = 0.781726
     59 VA = 0.935 cost = 0.638027
     60 VA = 0.939 cost = 0.674695
     61 VA = 0.939 cost = 0.556235
     62 VA = 0.935 cost = 0.765837
     63 VA = 0.937 cost = 0.870335
     64 VA = 0.940 cost = 0.735454
     65 VA = 0.941 cost = 0.571010
     66 VA = 0.941 cost = 0.707711
     67 VA = 0.939 cost = 0.721323
     68 VA = 0.939 cost = 0.651985
     69 VA = 0.941 cost = 0.730872
     70 VA = 0.942 cost = 0.660931
     71 VA = 0.945 cost = 0.840576
     72 VA = 0.945 cost = 0.619610
     73 VA = 0.940 cost = 0.624591
     74 VA = 0.944 cost = 0.657847
     75 VA = 0.945 cost = 0.588369
     76 VA = 0.944 cost = 0.602291
     77 VA = 0.946 cost = 0.631359
     78 VA = 0.946 cost = 0.764243
     79 VA = 0.947 cost = 0.697565
     80 VA = 0.948 cost = 0.698586
     81 VA = 0.945 cost = 0.686103
     82 VA = 0.946 cost = 0.765170
     83 VA = 0.946 cost = 0.588602
     84 VA = 0.946 cost = 0.652068
     85 VA = 0.949 cost = 0.643269
     86 VA = 0.945 cost = 0.708583
     87 VA = 0.947 cost = 0.742691
     88 VA = 0.946 cost = 0.601606
     89 VA = 0.951 cost = 0.643187
     90 VA = 0.949 cost = 0.654149
     91 VA = 0.950 cost = 0.700375
     92 VA = 0.948 cost = 0.698982
     93 VA = 0.949 cost = 0.734946
     94 VA = 0.951 cost = 0.515410
     95 VA = 0.951 cost = 0.697634
     96 VA = 0.948 cost = 0.622021
     97 VA = 0.946 cost = 0.704419
     98 VA = 0.951 cost = 0.591621
     99 VA = 0.947 cost = 0.602319
    100 VA = 0.947 cost = 0.636371
    101 VA = 0.949 cost = 0.617958
    102 VA = 0.950 cost = 0.609886
    103 VA = 0.952 cost = 0.570269
    104 VA = 0.949 cost = 0.681223
    105 VA = 0.952 cost = 0.536798
    106 VA = 0.951 cost = 0.626707
    107 VA = 0.950 cost = 0.701667
    108 VA = 0.952 cost = 0.585534
    109 VA = 0.949 cost = 0.675104
    110 VA = 0.953 cost = 0.631148
    111 VA = 0.953 cost = 0.740407
    112 VA = 0.951 cost = 0.658472
    113 VA = 0.951 cost = 0.585560
    114 VA = 0.954 cost = 0.687597
    115 VA = 0.952 cost = 0.711724
    116 VA = 0.954 cost = 0.593078
    117 VA = 0.952 cost = 0.682118
    118 VA = 0.956 cost = 0.690750
    119 VA = 0.954 cost = 0.714048
    120 VA = 0.955 cost = 0.707944
    121 VA = 0.954 cost = 0.556743
    122 VA = 0.954 cost = 0.614826
    123 VA = 0.956 cost = 0.582612
    124 VA = 0.953 cost = 0.535058
    125 VA = 0.953 cost = 0.579710
    126 VA = 0.953 cost = 0.519157
    127 VA = 0.954 cost = 0.655804
    128 VA = 0.952 cost = 0.614983
    129 VA = 0.954 cost = 0.562806
    130 VA = 0.954 cost = 0.712080
    131 VA = 0.953 cost = 0.562005
    132 VA = 0.954 cost = 0.585365
    133 VA = 0.951 cost = 0.669664
    134 VA = 0.952 cost = 0.615483
    135 VA = 0.954 cost = 0.757640
    136 VA = 0.954 cost = 0.729489
    137 VA = 0.954 cost = 0.687994
    138 VA = 0.953 cost = 0.550612
    139 VA = 0.952 cost = 0.639132
    140 VA = 0.955 cost = 0.553204
    141 VA = 0.953 cost = 0.615979
    142 VA = 0.956 cost = 0.615802
    143 VA = 0.954 cost = 0.738773
    144 VA = 0.956 cost = 0.643525
    145 VA = 0.956 cost = 0.709705
    146 VA = 0.959 cost = 0.597716
    147 VA = 0.956 cost = 0.628662
    148 VA = 0.956 cost = 0.616762
    149 VA = 0.957 cost = 0.574243
    150 VA = 0.955 cost = 0.606867
    151 VA = 0.956 cost = 0.640882
    152 VA = 0.954 cost = 0.719854
    153 VA = 0.958 cost = 0.522351
    154 VA = 0.954 cost = 0.658128
    155 VA = 0.954 cost = 0.597574
    156 VA = 0.955 cost = 0.616110
    157 VA = 0.955 cost = 0.541914
    158 VA = 0.956 cost = 0.679999
    159 VA = 0.953 cost = 0.559644
    160 VA = 0.958 cost = 0.565268
    161 VA = 0.957 cost = 0.568654
    162 VA = 0.958 cost = 0.629854
    163 VA = 0.956 cost = 0.634785
    164 VA = 0.959 cost = 0.589972
    165 VA = 0.957 cost = 0.556082
    166 VA = 0.957 cost = 0.545343
    167 VA = 0.955 cost = 0.649508
    168 VA = 0.956 cost = 0.669368
    169 VA = 0.957 cost = 0.754618
    170 VA = 0.959 cost = 0.641438
    171 VA = 0.954 cost = 0.548379
    172 VA = 0.959 cost = 0.627905
    173 VA = 0.954 cost = 0.585906
    174 VA = 0.955 cost = 0.536795
    175 VA = 0.955 cost = 0.611730
    176 VA = 0.958 cost = 0.616537
    177 VA = 0.958 cost = 0.687828
    178 VA = 0.957 cost = 0.557814
    179 VA = 0.957 cost = 0.646934
    Model saved



```python
# EVALUATE USING TEST DATA 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test_preprocessed, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Test Accuracy = 0.950



```python
###################################
#   Test the model on New Images #
#################################
```


```python
# LOAD THE NEW IMAGES FROM THE INTERNET TO A NUMPY ARRAY

path = 'data/'
image_name = ['0-20speed','1-30speed',
              '12-priority-road','13-yield',
              '14-stop','17-no-entry',
              '18-general-caution','3-60speed',
              '36-go-straight-right', 
              '40-roundabout-mandatory']

image_list = []

for name in image_name:
    img = imread(path + name + '.png')
    img = imresize(img, (32, 32))
    image_list.append(img)

own_set_x = np.array(image_list)
own_set_x = preprocess(own_set_x)
own_set_x = reshape(own_set_x)
own_set_y = np.array([0, 1, 12, 13, 14, 17, 18, 3, 36, 40])
print(own_set_x.shape, own_set_y.shape)
```

    (10, 32, 32, 1) (10,)



```python
#show selected image from internet 

number_of_images = len(image_list)
figure, axis = plt.subplots(1, number_of_images, figsize=(20, 20))
figure.subplots_adjust(hspace = .2, wspace=.001)
axis = axis.ravel()
    
for i in range(number_of_images):     
    image = image_list[i]
    axis[i].axis('off')
    axis[i].imshow(image)
```


![png](output_28_0.png)



```python
#CHECK HOW OUR SELECTED IMAGES FAIRED, AND ITS TOP 5 PREDICTION BASED on built-in top_k function 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    OUT = sess.run(tf.argmax(logits, 1), feed_dict={x: own_set_x, y: own_set_y, keep_prob: 1.0})
    print("", OUT, "<-predictions")
    print("", own_set_y, "<-actual")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    OUT = sess.run(tf.nn.top_k(tf.nn.softmax(logits), 5), feed_dict={x: own_set_x, y: own_set_y, keep_prob: 1.0})
    print(OUT[1].T)
    print("(top  5 predictions above) for each image")
    
print()    
print("probability for top 5 predictions for each image:")
for i in range(len(own_set_y)):
    print(i, OUT[0][i].T)


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(own_set_x, own_set_y )
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

     [ 0  1 12 13 14 17 18  3 36 40] <-predictions
     [ 0  1 12 13 14 17 18  3 36 40] <-actual
    [[ 0  1 12 13 14 17 18  3 36 40]
     [ 1  2 35  5  3 33 26  6 28 11]
     [ 6  6 25 12  8  9 27  1 38 16]
     [29 38 26 33  1 14 24 34 35  7]
     [34  0 40  2  4 40 36 11 12 17]]
    (top  5 predictions above) for each image
    
    probability for top 5 predictions for each image:
    0 [  9.32689548e-01   5.41089848e-02   1.31279845e-02   2.31403465e-05
       1.45926260e-05]
    1 [  9.99820888e-01   1.76206828e-04   1.00970851e-06   9.46819682e-07
       7.42028533e-07]
    2 [  1.00000000e+00   4.18683488e-09   7.02348402e-10   3.46118190e-10
       3.12163795e-10]
    3 [  1.00000000e+00   3.13466253e-10   2.64161998e-10   6.51305052e-11
       3.08685751e-11]
    4 [  9.99993563e-01   1.72575687e-06   1.65607423e-06   1.56471299e-06
       9.00445173e-07]
    5 [  1.00000000e+00   5.89572613e-10   1.53520793e-10   3.20257085e-11
       2.61226735e-11]
    6 [  1.00000000e+00   2.20382557e-09   8.86765050e-10   2.57165150e-10
       1.40842324e-10]
    7 [  9.99720633e-01   1.27843770e-04   7.89111946e-05   2.71247172e-05
       2.42004371e-05]
    8 [ 0.89542079  0.07540814  0.01817905  0.00890381  0.00110375]
    9 [  1.00000000e+00   5.24315036e-09   4.45303838e-09   3.62437103e-09
       2.18589946e-09]
    Test Accuracy = 1.000



```python

```
