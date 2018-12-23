import numpy as np
import cv2
from keras.models import Sequential
import csv
from os import listdir
from PIL import Image as PImage
from DataGenerator import DataGenerator


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
    		loadedImages.append(x)
    	except ValueError:
    		print ("Exception carregar la imatge")
    return loadedImages


if __name__ == '__main__':
	path = "/Users/lseijas/Desktop/TFG_Code/Image/Dataset/"
	# Parameters
	params = {'batch_size': 64,
	          'dim': (32,32,32),
	          'n_channels': 3,
	          'n_classes': 4}

	# Datasets
	labels = get_labelsID()
	partition = loadImages(path)
	

	# Generators
	training_generator = DataGenerator(partition['train'], labels, **params)
	validation_generator = DataGenerator(partition['validation'], labels, **params)

	# Design model
	#model = Sequential()
	#[...] # Architecture
	#model.compile()

# Train model on dataset
#model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=6)