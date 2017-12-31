import cv2
import numpy as np
import cPickle
import os
import sys
import time
import Tkinter
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from src.feature_extractors2 import SIFT_features,  SURF_features, descriptors_List2Array
from src.image_representation import BoW_hardAssignment, test_BoW_representation
from src.train import train_svm

start = time.time()

#Variables:
extractor = 'sift' # sift, DenseSIFT or surf
classifier = 'svm' # knn, rf, gnb, svm or lr
kernel_svm='rbf' #Kernel used in svm
n_features=300 #num. of key points detected with SIFT
k=512 #num. of words
C=1 #Penalty parameter C of the error term in svm algorithm
gamma=0.002 #kernel coefficient for 'rbf', 'poly', and 'sigmoid' in svm algorithm.

#Constants:
experiment_name = extractor + '_' + classifier + '_k' + str(k)+ '_C' + str(C) + '_gamma' + str(gamma)
experiment_filename = experiment_name + '.p'
predictions_filename = './predictions/' + experiment_name + '_predictions.p'

# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)


#Feature extractors:
if extractor == 'sift':
    #myextractor=(cv2.SIFT(nfeatures=300))
    myextractor=(cv2.xfeatures2d.SIFT_create(nfeatures = n_features))
    Train_descriptors, Train_label_per_descriptor = SIFT_features(myextractor, train_images_filenames, train_labels)
elif extractor == 'surf':
    #myextractor=(cv2.SURF(300))
    myextractor=(cv2.xfeatures2d.SURF_create(n_features))
    Train_descriptors, Train_label_per_descriptor = SURF_features(myextractor, train_images_filenames, train_labels)
else:
    sys.exit('[ERROR]: Not a valid extractor')
D=descriptors_List2Array(Train_descriptors)

#Getting BoVW with kMeans(Hard Assignment)
words, visual_words, codebook = BoW_hardAssignment(k, D, Train_descriptors)

# Train an SVM classifier.
clf, stdSlr=train_svm(visual_words, train_labels, experiment_filename, kernel_svm, C, gamma)


# get all the test data 
visual_words_test=test_BoW_representation(test_images_filenames, k, myextractor, codebook)

# Test the classification accuracy
print 'Testing the SVM classifier...'
init=time.time()
accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)
end=time.time()
print 'Done in '+str(end-init)+' secs.'
print 'Final accuracy: ' + str(accuracy)

end=time.time()
print 'Everything done in '+str(end-start)+' secs.'
### 69.02%
