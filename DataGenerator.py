import numpy as np
import keras
import csv

#New class generator for dataset A
class DataGenerator(keras.utils.Sequence):
  #We have 4 classes (forest fire, forest, city fire, city)
  def __init__(self, list_IDs, labels, batch_size, dim, n_channels,
               n_classes):
      'Initialization'
      # Podriem afegir el self.mean i self.std
      self.dim = dim
      self.batch_size = batch_size
      self.labels = labels
      self.list_IDs = list_IDs
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.shuffle = true
      #Ens barreja les mostres o no en funcio del paramtre shuffle 
      self.on_epoch_end()
      #
      self.sometimes = lambda aug: iaa.Sometimes(0.3, aug)

      self.seq = iaa.Sequential(
        [
          #Horizontally flip 50% of all images
          iaa.Fliplr(0.5), 
          #Crop some of the images by 0-10% of their height/width
          iaa.Crop(percent=(0, 0.05)),
          # Apply affine transformations to each image.
          # Scale/zoom them, translate/move them, rotate them and shear them.
          iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),shear=(-8, 8)),
          # Add gaussian noise.
          # For 50% of all images, we sample the noise once per pixel.
          # For the other 50% of all images, we sample the noise per pixel AND
          # channel. This can change the color (not only brightness) of the
          # pixels.
          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
          # Make some images brighter and some darker.
          # In 20% of all cases, we sample the multiplier once per channel,
          # which can end up changing the color of the images.
          iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ],
        #Do all of the above augmentations in random order
        random_order = True
      )

  #Ens dons el numero de un batch
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  #Funcio que s'executa per generar el batch que correspon al numero que hem obtingut de la funcio anterior
  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y

  #Aquesta funció ens permet canviar l'ordre d'exploració així els batches entre les epochs no s'assemblaran
  #Fer això eventualment farà que el nostre model sigui més robust
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    #Nomes es realitzara el shuffle si el parametre shuffle esta a true
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  # Aquesta funció rep una llista amb els ID dels target dels batch
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = np.load('data/' + ID + '.npy')

        # Store class
        y[i] = self.labels[ID]
    #Aquesta funcio converteix les nostres etiquetes numeriques guardades a 'y' a una forma binaria
    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


  