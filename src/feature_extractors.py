import cv2
import numpy as np
import h5py
import os
import time
"""
def SURF_features(SURFdetector, train_images_filenames, train_labels):
    #OUTDATED FROM SESSION 1
    if not os.path.exists('./src/descriptors/surf.h5'):
        # read the just 30 train images per class
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        Train_descriptors = []
        Train_label_per_descriptor = []

        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            if Train_label_per_descriptor.count(train_labels[i]) < 30:
                print 'Reading image ' + filename
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt, des = SURFdetector[0].detectAndCompute(gray, None)
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(train_labels[i])
                print str(len(kpt)) + ' extracted keypoints and descriptors'

        # Transform everything to numpy arrays

        D = Train_descriptors[0]
        L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

        for i in range(1, len(Train_descriptors)):
            D = np.vstack((D, Train_descriptors[i]))
            L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))
        f = h5py.File('./src/descriptors/surf.h5', 'w')
        f.create_dataset('D', data=D)
        f.create_dataset('L', data=L)
        f.close()
    else:
        sift_descriptor = h5py.File('./src/descriptors/surf.h5', 'r')
        D = np.array(sift_descriptor['D'])
        L = np.array(sift_descriptor['L'])
        sift_descriptor.close()
    return D, L"""

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
