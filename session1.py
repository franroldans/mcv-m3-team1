import cv2
import numpy as np
import cPickle
import os
import sys
import time
from src.evaluation import plot_confusion_matrix
from src.feature_extractors import SIFT_features
from src.train import train_knn

start = time.time()
#Variables:
extractor = 'sift'
classifier = 'knn'

#Constants:
experiment_name = extractor + '_' + classifier
experiment_filename = experiment_name + '.p'
predictions_filename = './predictions/' + experiment_name + '_predictions.p'
# read the train and test files
train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes '+str(set(train_labels))
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes '+str(set(test_labels))

#Feature extractors:
if extractor == 'sift':
    myextractor = cv2.SIFT(nfeatures=100)
else:
    sys.exit('[ERROR]: Not a valid extractor')

if not os.path.exists('./models/'+experiment_filename):
    D, L = SIFT_features(myextractor, train_images_filenames, train_labels)
    if classifier == 'knn':
        # Train a k-nn classifier
        myclassifier = train_knn(D, L, experiment_filename)
    else:
        sys.exit('[ERROR]: Not a valid classifier')
else:
    with open('./models/' + experiment_filename, 'r') as f:
        print 'Loading model from: {}'.format('./models/' + experiment_filename)
        myclassifier = cPickle.load(f)
        print 'Model loaded!'

# get all the test data and predict their labels
numtestimages = 0
numcorrect = 0
if not os.path.exists(predictions_filename):
    Y_pred = []
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = myextractor.detectAndCompute(gray, None)
        predictions = myclassifier.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        Y_pred.append(predictedclass)
        print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass
        numtestimages += 1
        if predictedclass == test_labels[i]:
            numcorrect += 1
    print 'Saving predictions in path: {}'.format(predictions_filename)
    cPickle.dump(Y_pred, open(predictions_filename, 'w'))
    print 'Predictions saved!'
else:
    with open(predictions_filename, 'r') as f:
        print 'Loading predictions from: {}'.format(predictions_filename)
        Y_pred = cPickle.load(f)
        print 'Predictions loaded'
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        print 'image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + Y_pred[i]
        numtestimages += 1
        if Y_pred[i] == test_labels[i]:
            numcorrect += 1

print 'Final accuracy: ' + str(numcorrect*100.0/numtestimages)
cm = plot_confusion_matrix(Y_pred, test_labels, experiment_name)
end = time.time()
print 'Done in '+str(end-start)+' secs.'

