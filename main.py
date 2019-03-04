import numpy as np
import os
from os import listdir
from skimage import io
from scipy import stats
from scipy.misc import imresize
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, Average 
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint

k_fold = 10
img_size = 28
epochs = 25


def get_img(data_path):
    # Getting image array from path:
    img = io.imread(data_path)
    img = imresize(img, (img_size, img_size, 1))
    return img

def get_dataset(dataset_path='/Data/Train'):
    # Getting all data from data path:
    # Getting labels
    labels = listdir(dataset_path) 
    labels.sort()
    print('Categories:\n', labels)
    len_datas = 0
    for label in labels:
        len_datas += len(listdir(dataset_path+'/'+label))
    X = np.zeros((len_datas, img_size, img_size), dtype='float32')
    Y = np.zeros(len_datas)
    count_data = 0
    # Encoding labels
    count_categori = [-1,''] 
    for label in labels:
        datas_path = dataset_path+'/'+label
        for data in listdir(datas_path):
            img = get_img(datas_path+'/'+data)
            # Store image array
            X[count_data] = img
            # Store image label
            if label != count_categori[1]:
                count_categori[0] += 1
                count_categori[1] = label
            Y[count_data] = count_categori[0]
            count_data += 1
    # Create dateset
    Y = keras.utils.to_categorical(Y)
    X = X.reshape(X.shape[0], img_size, img_size, 1)
    X = X.astype('float32')
    # Normalize data
    X /= 255
    return X, Y, labels

def get_testset(dataset_path='Test'):
    # Getting test set from data path:
    files = listdir(dataset_path) 
    files = [int(x.split(".")[0]) for x in files]
    files.sort()
    print(files[:5])
    len_datas = 14000
    X = np.zeros((len_datas, 28, 28), dtype='float32')
    count_data = 0
    # Convert image to array
    for file in files:
    	datas_path = dataset_path+'/'+ str(file) + '.jpg'
    	img = get_img(datas_path)
    	X[count_data] = img
    	count_data += 1
        
    X = X.reshape(X.shape[0], 28, 28, 1)
    X = X.astype('float32')
    X /= 255
    return X


def save_model(model):
    if not os.path.exists('/output/Model/'):
        os.makedirs('/output/Model/')

    model.save("/output/Model/model.h5")
    print('Model and weights saved')

def get_model(num_classes=40):
    model = Sequential()

    # Convulutional layer
    model.add(Conv2D(32, (3, 3), input_shape=(img_size,img_size,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    
    # Fully connected layer
    BatchNormalization()
    model.add(Dense(1024))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))

    model.add(Dense(num_classes,activation='softmax'))
    model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(X, Y):
    
    if not os.path.exists('/output/Model/'):
        os.makedirs('/output/Model/')
    
    # Perform k-fold cross validation
    sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.15, random_state=0)
    count = 0
    for train_index, test_index in sss.split(X, Y):
    	# Save best model
    	checkpoints = []
    	checkpoints.append(ModelCheckpoint('/output/Model/best_model{}.h5'.format(count), monitor='val_acc', verbose=0, save_best_only=True, mode='auto', period=1))
    	X_train, X_test = X[train_index], X[test_index]
    	Y_train, Y_test = Y[train_index], Y[test_index]
    	model = get_model()
    	model.fit(X_train, Y_train, batch_size=150, epochs=epochs, verbose=0, validation_data=(X_test, Y_test), callbacks=checkpoints)
    	count += 1


def ensemble(X_test,Y_test,labels,model_path='/output/Model/'):
	models= []
	for i in range(k_fold):
		model = load_model('/output/Model/best_model{}.h5'.format(i))
		models.append(model)
	
	final_pred = np.zeros((1, X_test.shape[0]))
	final_pred.fill(-1)
	for model in models:
		pred = model.predict_classes(X_test)
		pred = pred.reshape((1,X_test.shape[0]))
		final_pred = np.append(final_pred, pred, axis=0)
	print(final_pred[:10,:10])
	final_pred = stats.mode(final_pred)
	final_pred = final_pred[0].reshape((X_test.shape[0],))

	y_label = np.argmax(Y_test,axis=1)

	x = y_label[np.where(final_pred == y_label)]
	unique, counts = np.unique(x, return_counts=True)
	print(np.asarray((unique, counts)).T)

	num_correct = np.sum(final_pred == y_label)
	accuracy = num_correct/X_test.shape[0]
	print("Out of {}, we got {} right with accuracy {}".format(X_test.shape[0],num_correct,accuracy))
	"""
	with open('IWU_submission.txt','w') as output:
		files = listdir('Test_Data')
		for file in files:
			file_number = int(file)
			file_number -= 1
			label_index = final_pred[file_number]
			label_str = labels[label_index]
			output.write("{} {}".format(file,label_str))
	"""





if __name__ == '__main__':
	X, Y, labels = get_dataset()
	train_model(X,Y)

	#ensemble(X_test,Y_test,labels)

	
