import sys
from sklearn import decomposition
from sklearn.emsemble import RandomForestClassifier
from data_loader import load_data

# usage: python pca_randomforest.py dim
# example: python pca_randomforest.py 100

dim = int(sys.argv[1])

X = load_data('train_vali.txt')
Y = load_data('train_vali_label.txt').flatten()
X_test = load_data('test.txt')
Y_test_truth = load_data('test_label.txt').flatten()


# PCA dimension reduction
pca = decomposition.PCA(n_components = dim)
pca.fit(X)
X_reduct = pca.transform(X)
X_reduct_test = pca.transform(X_test)

# Random Forest classification
rf = RandomForestClassifier()
rf.fit(X_reduct, Y)
Y_test = rf.predict(X_reduct_test)

cnt = 0
total = len(Y_test_truth)
for i in range(total):
    if Y_test[i] == Y_test_truth[i]:
        cnt += 1


print 'correct prediction count: %d' % (cnt)
print 'total count: %d' % (total)
print 'pca_randomForest %d' % (dim)
print 'accuracy: %f' % (cnt/float(total))

