import sys
from sklearn import decomposition
from sklearn.svm import SVC
from data_loader import load_data

# usage: python pca_svm.py dim kernel_type
# example: python pca_svm.py 100 kernel

dim = int(sys.argv[1])
kernel_type = sys.argv[2]

X = load_data('train_vali.txt')
Y = load_data('train_vali_label.txt').flatten()
X_test = load_data('test.txt')
Y_test_truth = load_data('test_label.txt').flatten()


# PCA dimension reduction
# dim refers to the dimension
pca = decomposition.PCA(n_components = dim)
pca.fit(X)
X_reduct = pca.transform(X)
X_reduct_test = pca.transform(X_test)

# SVM classification with certain kernel functions
# kernel functions can be linear, poly, rbf, sigmoid
svm = SVC(kernel = kernel_type)
svm.fit(X_reduct, Y)
Y_test = svm.predict(X_reduct_test)

cnt = 0
total = len(Y_test_truth)
for i in range(total):
    if Y_test[i] == Y_test_truth[i]:
        cnt += 1


print 'correct prediction count: %d' % (cnt)
print 'total count: %d' % (total)
print 'pca_svm %d, kernel function %s' % (dim, kernel_type)
print 'accuracy: %f' % (cnt/float(total))

