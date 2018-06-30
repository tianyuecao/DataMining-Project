import sys
from sklearn import decomposition
from lr import logistic_regression
from data_loader import load_data

# usage: python pca_lr.py dim regularlizer
# example: python pca_lr.py 100 l1

dim = int(sys.argv[1])
regularizer = sys.argv[2]

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

# Logistic Regression classification with certain regularizer
# regularlizer parameter can be l1, l2
Y_test = logistic_regression(X_reduct, Y, regularizer, X_reduct_test)

cnt = 0
total = len(Y_test_truth)
for i in range(total):
    if Y_test[i] == Y_test_truth[i]:
        cnt += 1


print 'correct prediction count: %d' % (cnt)
print 'total count: %d' % (total)
print 'pca_logisticRegression %d, regularizer %s' % (dim, regularizer)
print 'accuracy: %f' % (cnt/float(total))

