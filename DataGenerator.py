import numpy as np
import keras
import csv
import imgaug as ia
from imgaug import augmenters as iaa


#New class generator for dataset A
class DataGenerator(keras.utils.Sequence):
  #We have 4 classes (forest fire, forest, city fire, city)
  def __init__(self, list_IDs, labels, batch_size, n_channels,
               n_classes):
      'Initialization'
      ia.seed(1)
      # Podriem afegir el self.mean i self.std
      self.dim = list_IDs.shape
      self.batch_size = batch_size
      self.labels = labels
      self.list_IDs = list_IDs
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.shuffle = True
      #Ens barreja les mostres o no en funcio del paramtre shuffle 
      self.on_epoch_end()
      print("He sortit del epoch end")
      #Posant 0.3 en contes de 0.5 fem mes debils els canvis aplicats
      self.sometimes = lambda aug: iaa.Sometimes(0.3, aug)
      print("He fixat la lambda del sometimes")
      self.seq = iaa.Sequential(
        [
          #Horizontally flip 50% of all images
          iaa.Fliplr(0.5), 
          #Vertically flip 20% of all images
          iaa.Flipud(0.2), 
          # crop some of the images by 0-10% of their height/width
          self.sometimes(iaa.Crop(percent=(0, 0.1))),
          # Apply affine transformations to some of the images
          # - scale to 80-120% of image height/width (each axis independently)
          # - translate by -20 to +20 relative to height/width (per axis)
          # - rotate by -45 to +45 degrees
          # - shear by -16 to +16 degrees
          # - order: use nearest neighbour or bilinear interpolation (fast)
          # - mode: use any available mode to fill newly created pixels
          #         see API or scikit-image for which modes are available
          # - cval: if the mode is constant, then use a random brightness
          #         for the newly created pixels (e.g. sometimes black,
          #         sometimes white)
          iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),shear=(-8, 8),order=[0, 1], cval=(0, 255)),
          #Add gaussian noise.
          #For 50% of all images, we sample the noise once per pixel.
          #For the other 50% of all images, we sample the noise per pixel AND
          #channel. This can change the color (not only brightness) of the
          #pixels.
          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
          #Execute 0 to 5 of the following (less important) augmenters per
          #image. Don't execute all of them, as that would often be way too
          #strong.
          iaa.SomeOf((0, 5),
              [
                  #Convert some images into their superpixel representation,
                  #sample between 20 and 200 superpixels per image, but do
                  #not replace all superpixels with their average, only
                  #some of them (p_replace).
                  self.sometimes(
                      iaa.Superpixels(
                          p_replace=(0, 1.0),
                          n_segments=(20, 200)
                      )
                  ),

                  #Blur each image with varying strength using
                  #gaussian blur (sigma between 0 and 3.0),
                  #average/uniform blur (kernel size between 2x2 and 7x7)
                  #median blur (kernel size between 3x3 and 11x11).
                  iaa.OneOf([
                      iaa.GaussianBlur((0, 3.0)),
                      iaa.AverageBlur(k=(2, 7)),
                      iaa.MedianBlur(k=(3, 11)),
                  ]),

                  #Sharpen each image, overlay the result with the original
                  #image using an alpha between 0 (no sharpening) and 1
                  #(full sharpening effect).
                  iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                  #Same as sharpen, but for an embossing effect.
                  iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                  #Search in some images either for all edges or for
                  #directed edges. These edges are then marked in a black
                  #and white image and overlayed with the original image
                  #using an alpha of 0 to 0.7.
                  self.sometimes(iaa.OneOf([
                      iaa.EdgeDetect(alpha=(0, 0.7)),
                      iaa.DirectedEdgeDetect(
                          alpha=(0, 0.7), direction=(0.0, 1.0)
                      ),
                  ])),

                  #Add gaussian noise to some images.
                  #In 50% of these cases, the noise is randomly sampled per
                  #channel and pixel.
                  #In the other 50% of all cases it is sampled once per
                  #pixel (i.e. brightness change).
                  iaa.AdditiveGaussianNoise(
                      loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                  ),

                  #Either drop randomly 1 to 10% of all pixels (i.e. set
                  #them to black) or drop them on an image with 2-5% percent
                  #of the original size, leading to large dropped
                  #rectangles.
                  iaa.OneOf([
                      iaa.Dropout((0.01, 0.1), per_channel=0.5),
                      iaa.CoarseDropout(
                          (0.03, 0.15), size_percent=(0.02, 0.05),
                          per_channel=0.2
                      ),
                  ]),

                  #Invert each image's chanell with 5% probability.
                  #This sets each pixel value v to 255-v.
                  iaa.Invert(0.05, per_channel=True), # invert color channels

                  #Add a value of -10 to 10 to each pixel.
                  iaa.Add((-10, 10), per_channel=0.5),

                  #Change brightness of images (50-150% of original value).
                  iaa.Multiply((0.5, 1.5), per_channel=0.5),

                  #Improve or worsen the contrast of images.
                  iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                  #Convert each image to grayscale and then overlay the
                  #result with the original with random alpha. I.e. remove
                  #colors with varying strengths.
                  iaa.Grayscale(alpha=(0.0, 1.0)),

                  #In some images move pixels locally around (with random strengths).
                  self.sometimes(
                      iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                  ),

                  #In some images distort local areas with varying strength.
                  self.sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
              ],
              #Do all of the above augmentations in random order
              random_order = True
            )
          ],
        #Do all of the above augmentations in random order
        random_order = True
      )
      print("He sortit del sequential")

  #Ens dons el numero de un batch
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  #Funcio que s'executa per generar el batch que correspon al numero que hem obtingut de la funcio anterior
  def __getitem__(self, index):
    'Generate one batch of data'
    print("Index: " + str(index))
    print("Batch: " + str(self.batch_size))
    print("Self.ids shape: "+ str(self.list_IDs.shape))
    # Generate indexes of the batch
    indexes = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]
    print("Indexes: " + str(indexes))
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    print("List elements: " + str(list_IDs_temp))
    
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

