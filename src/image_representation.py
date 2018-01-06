import cv2
import cPickle
import numpy as np
import time
import os
import math
import scipy.cluster.vq as vq
from sklearn import cluster


def build_spatial_pyramid(descriptor, level):

    DSIFT_STEP_SIZE = 4
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    #h=20
    #w=15
    h = 256 / step_size
    w = 256 / step_size
    idx_crop = np.resize(np.array(range(len(descriptor))), [h, w])
    #idx_crop = np.array(range(len(descriptor)))
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height/bh, width/bw, bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
            idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid

def spatial_pyramid_matching(descriptor, codebook, level, ids, k):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(descriptor, 0)
        code = [input_vector_encoder(crop, codebook, ids, k) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(descriptor, 0)
        pyramid += build_spatial_pyramid(descriptor, 1)
        code = [input_vector_encoder(crop, codebook, ids, k) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(descriptor, 0)
        pyramid += build_spatial_pyramid(descriptor, 1)
        pyramid += build_spatial_pyramid(descriptor, 2)
        code = [input_vector_encoder(crop, codebook, ids, k) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))

def input_vector_encoder(feature, codebook, ids, k):

   """ #words = codebook.predict([feature])
    print ids.max()
   #print words.shape
    word_hist = np.array([np.bincount(feature[ids == i], minlength=k) for i in
                            range(0, ids.max()+1)], dtype=np.float64)
    return word_hist"""
   code, _ = vq.vq(feature, codebook)
   word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
   return word_hist

def BoW_hardAssignment(k, D, ids, spatial_pyramid=False, save_model=True):
    if not os.path.exists('./models/' + 'codebook.p'):

        # compute the codebook
        print 'Computing kmeans with ' + str(k) + ' centroids'
        init = time.time()
        codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                           reassignment_ratio = 10 ** -4, random_state = 42)
        codebook.fit(D)
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.'

        # get train visual word encoding
        print 'Getting Train BoVW representation'
        init = time.time()
        words = codebook.predict(D)
        if spatial_pyramid:
            pass
            #visual_words = build_pyramid(words, ids, k)
        else:
            visual_words = np.array([np.bincount(words[ids == i], minlength=k) for i in
                                range(0, ids.max() + 1)], dtype=np.float64)

        if save_model:
            print 'Saving Codebook...'
            cPickle.dump(codebook, open('./models/' + 'codebook.p', 'w'))
            print 'Model saved!'
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.'
    else:
        with open('./models/' + 'codebook.p', 'r') as f:
            print 'Loading model from: {}'.format('./models/' + 'codebook.p')
            codebook = cPickle.load(f)
            words = codebook.predict(D)
            visual_words = np.array([np.bincount(words[ids == i], minlength=k) for i in
                                     range(0, ids.max() + 1)], dtype=np.float64)
            print 'Model loaded!'

    return words, visual_words, codebook

def test_BoW_representation(test_images_filenames, k, myextractor, codebook):

    print 'Getting Test BoVW representation'
    init=time.time()
    visual_words_test=np.zeros((len(test_images_filenames),k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename=test_images_filenames[i]
        print 'Reading image '+filename
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt,des=myextractor.detectAndCompute(gray,None)
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return visual_words_test

def build_pyramid(prediction, descriptors_indices, k=512):
    levels = [[1, 1], [2, 2], [4, 4]]
    v_words = []

    # Build representation for each image
    for i in range(0, descriptors_indices.max() + 1):
        try:
            image_predictions = prediction[descriptors_indices[0:-1] == i]
            print image_predictions.shape
            keypoints_shape = map(int, [image_predictions.shape[0], image_predictions.shape[1]])
            kp_i = keypoints_shape[0]
            kp_j = keypoints_shape[1]
            image_predictions_grid = np.reshape(image_predictions, keypoints_shape)

            im_representation = []

            for level in range(0, len(levels)):
                num_rows = levels[level][0]
                num_cols = levels[level][1]
                step_i = int(math.ceil(float(kp_i) / float(num_rows)))
                step_j = int(math.ceil(float(kp_j) / float(num_cols)))
                print step_i
                print step_j
                print num_rows
                print num_cols
                for i in range(0, kp_i, step_i):
                    for j in range(0, kp_j, step_j):
                        try:
                            hist = np.array(np.bincount(image_predictions_grid[i:i + step_i, j:j + step_j].reshape(-1).astype(int),
                                                        minlength=k))
                            word_hist, bin_edges = np.histogram(image_predictions_grid, bins=k, normed=True)
                            im_representation = np.hstack((im_representation, word_hist))
                        except:
                            pass

            v_words.append(im_representation)
        except IndexError:
            pass
    return np.array(v_words, dtype=np.float64)

