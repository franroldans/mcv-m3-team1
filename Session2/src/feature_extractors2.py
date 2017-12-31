import cv2
import numpy as np
import h5py
import os
import time

def SIFT_features(SIFTdetector, train_images_filenames, train_labels):
    if not os.path.exists('./src/descriptors/sift.npy'):
        print 'Computing SIFT features...'
        init=time.time()
        Train_descriptors = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            #print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            Train_descriptors.append(des)

        Train_descriptors_array = np.asarray(Train_descriptors)

        np.save('./src/descriptors/sift',Train_descriptors_array)
        end=time.time()
        print 'Done in '+str(end-init)+' secs.'
    else:
        print 'Loading SIFT features...'
        init=time.time()
        Train_descriptors_array = np.load('./src/descriptors/sift.npy')
        end=time.time()
        print 'Done in '+str(end-init)+' secs.'
    return Train_descriptors_array

def SURF_features(SURFdetector, train_images_filenames, train_labels):

    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):
        filename = train_images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SURFdetector.detectAndCompute(gray, None)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print str(len(kpt)) + ' extracted keypoints and descriptors'

    return Train_descriptors, Train_label_per_descriptor

def descriptors_List2Array(descriptors):

    size_descriptors=descriptors[0].shape[1]
    D=np.zeros((np.sum([len(p) for p in descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint=0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint+=len(descriptors[i])
    return D

