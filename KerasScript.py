import numpy as np
import cv2 as cv
from keras.models import Sequential
import csv
from os import listdir
from PIL import Image as PImage
from DataGenerator import DataGenerator
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def get_labelsID():
	labels = []
	with open('data.csv', 'r') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line_count = 0
	    for row in csv_reader:
        	if line_count == 0:
        		line_count = 1
	        else:
	            #Llegim les columnes que ens proporcionen la informacio sobre el resultat de la imatge
	            x = np.array([row[1], row[2], row[3], row[4]])
	            x = x.astype('float32')
	            #Afegim la etiqueta de la imatge al array d'etiquetes.
	            labels.append(x)
	return labels

def loadImages(path):
	# return array of images
	imagesList = listdir(path)
	loadedImages = []
	for image in imagesList:
		try:
			if (path + image) == path + '.DS_Store':
				continue
			img = PImage.open(path + image)
			x = np.asarray(img)
			#Descartem les imatges que son en blanc i negre
			if x.shape[-1] == 3:
				#Escalem la imatge a 32x32x3
				img_resize = cv.resize(x, (32,32))
				#Normalitzem les imatges de 0 a 255
				img_norm = cv.normalize(img_resize, None, 0, 255, cv.NORM_MINMAX)
				#Normalize inputs from 0-255 to 0.0-1.0
				#img_norm = img_norm.astype('float32')
				#img_norm = img_norm / 255.0
				loadedImages.append(img_norm)
		except ValueError:
			print ("Exception carregar la imatge")
	return loadedImages

def dataSeparation(data, labels, train = 0.8):
	#Shuffle data tant de la informació com de les etiquetes
	data, labels = shuffle(data, labels, random_state = 0)

	#Agafem un 80% d'imatges de la carpeta data, en concret, el 80% de les primeres imatges
	X_train = data[(int)(0): (int)(train*len(data))]
	Y_train = labels[(int)(0): (int)(train*len(labels))]	

	#Agafem el 20% d'imatges de la carpeta data que no hem guardat ni al X_train
	X_test = data[(int)(train*len(data)):len(data)]
	Y_test = labels[(int)(train*len(data)):len(labels)]

	return X_test, X_train, Y_test, Y_train

def dataValidation(X_train, Y_train, val = 0.2):

	#Agafem el 20% d'imatges del X_train
	X_val = X_train[(int)(0) : (int)(val*len(X_train))]
	Y_val = Y_train[(int)(0) : (int)(val*len(Y_train))]

	#Actualitzem el X_train er eliminar el 20% d'imatges que hem guardat en el X_val
	X_train = X_train[(int)(val*len(X_train)): (int)(len(X_train))]
	Y_train = Y_train[(int)(val*len(Y_train)): (int)(len(Y_train))]

	return X_val, Y_val, X_train, Y_train 

def arquitectureModel(x_train, num_classes = 4):
	weight_decay = 1e-4
	model = Sequential()
	model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	 
	model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.3))
	 
	model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.4))
	 
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	 
	model.summary()

	return model

if __name__ == '__main__':
	path = "/Users/lseijas/Desktop/TFG_Code/Image/Dataset/"

	# Datasets
	labels = get_labelsID()
	labels = np.asarray(labels)
	data = loadImages(path)
	data = np.asarray(data)



	X_test, X_train, Y_test, Y_train = dataSeparation(data, labels)

	mean = np.mean(X_train,axis=(0,1,2,3))
	std = np.std(X_train,axis=(0,1,2,3))
	X_train = (X_train-mean)/(std+1e-7)
	X_test = (X_test-mean)/(std+1e-7)

	X_val, Y_val, X_train, Y_train = dataValidation(X_train, Y_train)

	#data augmentation
	datagen = ImageDataGenerator(
	    rotation_range=15,
	    width_shift_range=0.1,
	    height_shift_range=0.1,
	    horizontal_flip=True,
	    )
	datagen.fit(X_train)
	model = arquitectureModel(X_train)

	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
	lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='auto', min_lr=10e-7)
	callback_list = [early_stopping, lr_schedule]

	opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),\
	                    steps_per_epoch=X_train.shape[0] // 32,epochs=125,\
	                    verbose=1,validation_data=(X_val,Y_val),callbacks=callback_list)
	#save to disk
	model.save_weights('model.h5') 
	
	#testing
	scores = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
	print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
