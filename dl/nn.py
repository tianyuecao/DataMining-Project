from data_util import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras import regularizers

# pca feature dimension
ndim = 100
if len(sys.argv) != 1:
    ndim = int(sys.argv[1])
# label dimension
ldim = 43
batch_size = 32
nb_epoch = 30
dropout_rate = 0.5
hidden_width = 256

train = DataLoader(label_dim=ldim, normalize=True, pca=True, pca_dim=ndim, istrain=True,
                   datapath="./data/train.txt", labelpath="./data/train_label.txt")
vali = DataLoader(label_dim=ldim, normalize=True, pca=True, pca_dim=ndim, istrain=False,
                  datapath="./data/vali.txt", labelpath="./data/vali_label.txt")
test = DataLoader(label_dim=ldim, normalize=True, pca=True, pca_dim=ndim, istrain=False,
                  datapath="./data/test.txt", labelpath="./data/test_label.txt")

print "Network building begin"
model = Sequential()
model.add(Dense(hidden_width, input_shape=(ndim,), activity_regularizer=regularizers.l2(1e-4), activation='sigmoid', kernel_initializer='glorot_normal'))
model.add(Dropout(dropout_rate))
model.add(Dense(hidden_width, activation='tanh', kernel_initializer='glorot_normal'))
model.add(Dropout(dropout_rate))
model.add(Dense(hidden_width, activation='tanh', kernel_initializer='glorot_normal'))
model.add(Dropout(dropout_rate))
model.add(Dense(ldim, activation='softmax', kernel_initializer='glorot_normal'))

# print model structure
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print "Network training begin"
history = model.fit(train.data, train.label, batch_size=batch_size,
                    nb_epoch=nb_epoch, verbose=2, validation_data=(vali.data, vali.label),
                    callbacks=[TensorBoard(log_dir='./tmp/log')])

score = model.evaluate(test.data, test.label, verbose=1)
print "\nTest accuracy: ", score[1]
