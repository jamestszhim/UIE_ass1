import numpy as np
from abc import ABCMeta, abstractmethod
from skimage.color import rgb2gray
from skimage import feature
import matplotlib.pyplot as plt

class Feature():
  """
  Feature base class.
  """
  __metaClass__ = ABCMeta

  @abstractmethod
  def evaluate(self, x1, x2):
    pass


# Feature classes
class SI(Feature):
  
  def evaluate(self, img, widthPadding=10):
    img = img[1]
    
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
        
    square = np.zeros((5, 5))
    square[2, 2] = 4
    s = feature.shape_index(square, sigma=0.1)
    
    return s.flatten()
        
    
class HOG(Feature):

  def evaluate(self, img, orientations=8, pixels_per_cell=(16,16), cells_per_block=(4,4), widthPadding=10):
    img = img[0]
      
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    hog_features = feature.hog(img, orientations, pixels_per_cell, cells_per_block, visualise=False)
    
    #plt.imshow(hog_image)
    
    return hog_features

class Daisy(Feature):

  def evaluate(self, img, orientations=8, pixels_per_cell=(16,16), cells_per_block=(4,4), widthPadding=10):
    img = img[1]
    
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    daisy_features = feature.daisy(img, step=7 , radius=9, rings=10, histograms=4, orientations=8, normalization='l1', sigmas=None, ring_radii=None, visualize=False)
    
    #plt.imshow(daisy_image)
    
    return daisy_features.flatten()

class Pixel(Feature):

  def evaluate(self, img, widthPadding=10):    
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    pixel = img[1].flatten()
    
    #plt.imshow(pixel)
    
    return pixel

class Edge(Feature):

  def evaluate(self, img, widthPadding=10):
    img = img[0]
    
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    edges = feature.canny(img, sigma=3)
    
    #plt.imshow(edges)
    
    return edges.flatten()
  
class Identity(Feature):

  def evaluate(self, x1, x2):
    return 1
#add some comments
