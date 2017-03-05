
### In this project, I have used what I've learned about deep neural networks and convolutional neural networks to classify german traffic signs using the TensorFlow framework.  This project is submitted as partial fulfillment of the requirements in order to pass the first term of Udacity's self-driving car engineer nano degree program.

*Jupyter Notebook*
- https://github.com/mithi/self-driving-project-2/blob/master/submission/Traffic_Sign_Classifier.ipynb

*Generated HTML*
- https://github.com/mithi/self-driving-project-2/blob/master/submission/report.html

*PDF Writeup*
- https://github.com/mithi/self-driving-project-2/blob/master/submission/WRITEUP.pdf

### Recommendations

- To strengthen the predictions of this convolutional neural network, I think we should feed it more data. Some of the classes were represented far more than others. The lack of balance in the training data set results in a bias towards classes with more data points. We can generate "fake" data points for less represented classes by applying small but random translational and rotational shifts as well as shearing and warping.

- Preprocessing the data can be made more faster by using better localized histogram equalization techniques and also no longer normalizing the values to be floats within the range of 0 to 1. Using integers between 0, 255 might be sufficient.

- Visualizing the networks weights can also help in designing the achitecture. Visualize them by plotting the weights filters of the convolutional layers as grayscale images

- Check the data points which are incorrectly predicted by the system and try to analyze this information

- Experiment with hyperparameters and other architectures

### Miscellaneous

This convolutional neural network is a modified version of this code as presented in the lectures:
- https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

Datasets used publicly available here
- http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset 
- https://d17h27t6h515a5.cloudfront.et/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip

### Good Readings

- http://cs231n.github.io/
- http://www.deeplearningbook.org/
- http://neuralnetworksanddeeplearning.com/
- http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html
- http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
- https://github.com/hengck23-udacity/udacity-driverless-car-nd-p2
- http://navoshta.com/traffic-signs-classification/




