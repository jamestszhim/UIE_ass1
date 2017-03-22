import numpy as np


class ProgrammaticError(Exception):
  """Exception raised when method gets called at a wrong time instance.

  Attributes:
      msg  -- The error message to be displayed.
  """

  def __init__(self, msg):
    self.msg = msg
    print("\033[91mERROR: \x1b[0m {}".format(msg))


class SVM_Model():
  
  feature_vec = []
  fitting_done = False

  def __init__(self):
    self.feature_vec = []
    self.fitting_done = False


  def set_feature_vector(self, feature_vec):
    self.feature_vec = feature_vec

  def compute_feature_matrix(self, input_data):
    n_samples = input_data.shape[0]
    n_features = len(self.feature_vec)
    n_dimension = 0
    
    for i in range(0, n_features):
        n_dimension = n_dimension + self.feature_vec[i].evaluate(input_data[0]).shape[0]
    
    feature_matrix = np.zeros([n_samples, n_dimension])
    for i in range(0, n_samples):
        j = 0
        for k in range(0, n_features):
            feature = self.feature_vec[k].evaluate(input_data[i])
            feature_matrix[i][j : feature.shape[0] + j] = feature
            j = feature.shape[0]
    return feature_matrix

       
       