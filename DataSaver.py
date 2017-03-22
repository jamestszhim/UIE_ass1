# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import os


class DataSaver():

  directory = ''
  filename = ''
  path = ''

  def __init__(self, directory, filename):
    self.directory = directory
    self.filename = filename
    self.path = os.path.join(self.directory, self.filename)
    if not os.path.exists(self.directory):
      os.mkdir(self.directory)

  def save_to_file(self, input_data, output_data):
    data = np.hstack([input_data, output_data])
    print('Saving data to relative path {}'.format(self.path))
    pkl.dump(data, open(self.path, 'wb'))

  def restore_from_file(self):
    training_data = pkl.load(open(self.path, "rb"))
    print(training_data.keys())
   
    return training_data

