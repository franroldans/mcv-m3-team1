import cv2
import numpy as np
import cPickle
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from features.feature_extractors import SIFT_features
start = time.time()
# read the train and test files
extractor = 'sift'
classifier = 'knn'
experiment_filename = extractor + '_' + classifier + '.p'
train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))

print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes '+str(set(train_labels))
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes '+str(set(test_labels))
SIFTdetector = cv2.SIFT(nfeatures=100)
if not os.path.exists('./models/'+experiment_filename):
    D, L = SIFT_features(SIFTdetector, train_images_filenames, train_labels)
    # Train a k-nn classifier
    print 'Training the knn classifier...'
    myknn = KNeighborsClassifier(n_neighbors=5)
    myknn.fit(D, L)
    print 'Done!'
    print 'Saving KNN model'
    cPickle.dump(myknn, open('./models/'+experiment_filename, 'w'))
    print 'KNN model saved!'
else:
    with open('./models/' + experiment_filename, 'r') as f:
        print 'Loading model from: {}'.format('./models/' + experiment_filename)
        myknn = cPickle.load(f)
        print 'Model saved!'
# get all the test data and predict their labels

numtestimages=0
numcorrect=0
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    kpt, des=SIFTdetector.detectAndCompute(gray,None)
    predictions = myknn.predict(des)
    values, counts = np.unique(predictions, return_counts=True)
    predictedclass = values[np.argmax(counts)]
    print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass
    numtestimages += 1
    if predictedclass == test_labels[i]:
        numcorrect += 1

print 'Final accuracy: ' + str(numcorrect*100.0/numtestimages)

end=time.time()
print 'Done in '+str(end-start)+' secs.'

## 30.48% in 302 secs.