# Deep-Learning-Final-Project
Deep Learning Final Project: Built and trained Stacked CNN AutoEncoder and Deep CNN AutoEncoder based on STL-10 dataset 

## Problem Description
The main goal of our project is to use unlabeled images to improve the classification accuracy of a convolutional neural network. In practical applications, the amount of available unlabeled data is larger in size than that of the labeled data. The natural question that follows is: how unlabeled data can be used to pretrain a neural network in order to improve its classification\regression accuracy. To achieve our goal, we experiment the possibility of mixing convolutional networks with autoencoders.

## Dataset
We experiment on a dataset (STL-10) that has been studied in the literature since 2011. The data consists of 5,000 labeled images for training, 8,000 labeled images for testing, and 100,000 unlabeled images. Each image has a size of 96x96 pixels in color and represents one of 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). See details of STL-10 in https://cs.stanford.edu/~acoates/stl10/.

## Convolutional AutoEncoder
We implemented a Deconvolutional Layer to reverse the process of convolution in keras using Theano. See details in AutoEncoderLayers.py. Through training Convolutional AutoEncoder model layer by layer, we get the feature maps layer by layer, and visualized them in testModels.py

* See details in our report (Convolutional AutoEncoders - Report.docx)

