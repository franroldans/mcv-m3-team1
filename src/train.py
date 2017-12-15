import cPickle
from sklearn.neighbors import KNeighborsClassifier


def train_knn(descriptors, labels, experiment_filename):
    myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    myknn.fit(descriptors, labels)
    print 'Done!'
    print 'Saving KNN model...'
    cPickle.dump(myknn, open('./models/' + experiment_filename, 'w'))
    print 'KNN model saved!'
    return myknn
