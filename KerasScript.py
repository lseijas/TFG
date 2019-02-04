# ********************************************************************************+
# @Author: Laia Seijas
# @Goal: Classify images.
# @Date: 31/12/2018
# *********************************************************************************
import numpy as np
import cv2 as cv
from keras.models import Sequential
import csv
from os import listdir
from PIL import Image as PImage
from DataGenerator import DataGenerator
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from pickle import load
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

def descompressOutput():
	container = np.load('data.npz')
	data = [container[key] for key in container]
	container = np.load('labels.npz')
	labels = [container[key] for key in container]
	print (data)
	print (labels)
	return data, labels

def compressOutput(data, labels):
	np.savez('data.npz', *data)
	np.savez('labels.npz', *labels)

def make_mosaic(imgs, nrows, ncols, border=1):
	nimgs = imgs.shape[0]
	imshape = imgs.shape[1:]
	mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border, ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)
	paddedh = imshape[0] + border
	paddedw = imshape[1] + border
	for i in range(0, nimgs):
		row = int(np.floor(i / ncols))
		col = i % ncols
		mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i]
	return mosaic

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
	"""Wrapper around pl.imshow"""
	if cmap is None:
	    cmap = cm.jet
	if vmin is None:
	    vmin = data.min()
	if vmax is None:
	    vmax = data.max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
	plt.colorbar(im, cax=cax)
	#plt.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1))
	
def displayWeights(model, layer):
	# Visualize weights 
	W = model.layers[layer].get_weights()[0]
	W = np.squeeze(W)
	if len(W.shape) == 4:
		W = W.reshape((-1,W.shape[2],W.shape[3]))
	print("W shape : ", W.shape)
	plt.figure(figsize=(15, 15))
	plt.title('conv weights')
	s = int(np.sqrt(W.shape[0])+1)
	nice_imshow(plt.gca(), make_mosaic(W, s, s), cmap=plt.cm.binary)
	plt.show()

#Funcion para muestrear la accuracy y la loss del modelo
def showGraphics(history): 

	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()

	plt.plot(history.history['lr'])
	plt.title('model lr')
	plt.ylabel('lr')
	plt.xlabel('epoch')
	plt.legend(['lr'], loc='upper left')
	plt.show()

def get_label(image):
	with open('data.csv', 'r') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line_count = 0
	    for row in csv_reader:
        	if line_count == 0:
        		line_count = 1
	        else:
	        	#print("row:"+ str(row[0]))
	        	if row[0] == image:
		        	x = np.array([row[1], row[2], row[3], row[4]])
		        	x = x.astype('float32')
		        	return x
		        line_count = line_count + 1
	print("Label not found")
	y = np.array([0, 0, 0, 0])
	y = y.astype('float32')
	return y

def loadImages(path):
	# return array of images
	imagesList = listdir(path)
	loadedImages = []
	labels = []
	for image in imagesList:
		try:
			if (path + image) == path + '.DS_Store':
				continue
			try:	
				img = PImage.open(path + image)
				x = np.asarray(img)
				#Descartem les imatges que son en blanc i negre
				if x.shape[-1] == 3:
					im, ext = image.split(".")
					print (im)
					label = get_label(im)
					print(label)
					labels.append(label)
					#Escalem la imatge a 32x32x3
					img_resize = cv.resize(x, (32,32))
					#Normalitzem les imatges de 0 a 255
					img_norm = cv.normalize(img_resize, None, 0, 255, cv.NORM_MINMAX)
					loadedImages.append(img_norm)
			except OSError:
				print ("Exception al obrir la imatge")
		except ValueError:
			print ("Exception carregar la imatge")
	return loadedImages, labels

def dataSeparation(data, labels, train = 0.8):
	#Shuffle data tant de la informació com de les etiquetes
	print(data.shape)
	print(labels.shape)
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

def showAndSave(X_train, Y_train):
	# configure batch size and retrieve one batch of images
	os.makedirs('images', exist_ok=True)
	for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=9, save_to_dir='images', save_prefix='aug', save_format='png'):
		# create a grid of 3x3 images
		print(X_batch.shape)
		print(Y_batch.shape)
		for i in range(0, 9):
			print (Y_batch[i])
			pyplot.subplot(330 + 1 + i)
			pyplot.imshow(X_batch[i], cmap=pyplot.get_cmap('gray'))
		# show the plot
		pyplot.show()
		break

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
	#data, labels = descompressOutput()
	
	data, labels = loadImages(path)
	data = np.asarray(data)
	labels = np.asarray(labels)

	X_test, X_train, Y_test, Y_train = dataSeparation(data, labels)

	mean = np.mean(X_train,axis=(0,1,2,3))
	std = np.std(X_train,axis=(0,1,2,3))
	#Apliquem la Z-score normalization aixi aconseguim "linearly transformed 
	#data values having a mean of zero and a standard deviation of 1"
	X_train = (X_train-mean)/(std+1e-7)
	X_test = (X_test-mean)/(std+1e-7)

	X_val, Y_val, X_train, Y_train = dataValidation(X_train, Y_train)

	#compressOutput(data, labels)

	X_train2 = DataGenerator(X_train, Y_train)
	

	#data augmentation
	datagen = ImageDataGenerator(
	    rotation_range=15,
	    width_shift_range=0.1,
	    height_shift_range=0.1,
	    horizontal_flip=True,
	    )
	datagen.fit(X_train)

	#showAndSave(X_train, Y_train)

	model = arquitectureModel(X_train)

	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
	lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='auto', min_lr=10e-7)
	#callback_list = [early_stopping, lr_schedule]
	callback_list = [lr_schedule]

	opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

	model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
	
	history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),\
	                    steps_per_epoch=X_train.shape[0] // 32,epochs=120,\
	                    verbose=1,validation_data=(X_val,Y_val),callbacks=callback_list)

	displayWeights(model, 0)

	#save to disk
	#model.save_weights('model.h5')
	
	#show graphics: the accuracy, loss and learning rate
	showGraphics(history)
	
	#testing
	scores = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
	print('\nTest result:\n accuracy: %.3f loss: %.3f\n' % (scores[1]*100,scores[0]))

	Y_pred = model.predict(X_test)
	# Convert predictions classes to one hot vectors 
	Y_pred_classes = np.argmax(Y_pred,axis = 1) 
	# Convert validation observations to one hot vectors
	Y_true = np.argmax(Y_test,axis = 1) 
	target_names = ['Incendi Forestal', 'Incendi Ciutat', 'Bosc', 'Edifici']
	print(classification_report(Y_true, Y_pred_classes, target_names=target_names))

