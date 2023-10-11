#Library in Outlier Handling Class
#Multivariate
from sklearn.neighbors import LocalOutlierFactor

class Mutivariate_Outliers():

  """
  Remove multivariate outliers by suitable method

  Input:
  ------
  Data_train, Data_test is removed univariate outliers.

  Returns:
  --------
  Cleaned Data_train and Test

  """
  def __init__(self, data_train, data_test):
        self.data_train_0 = data_train
        self.data_test_0 = data_test
        self.LOF()

    # 1. LOF
  def LOF(self):
        self.data_train_LOF = self.data_train_0.copy()
        self.data_test_LOF = self.data_test_0.copy()
        while True:
            try:
                self.n_neighbors = 20
                break
            except:
                print("Error values!")
        LOF = LocalOutlierFactor(n_neighbors = self.n_neighbors)
        LOF.fit(self.data_train_LOF)
        self.Outlier_LOF = self.data_train_LOF[LOF.fit_predict(self.data_train_LOF) == -1]
        self.Data_train_LOF = self.data_train_LOF[LOF.fit_predict(self.data_train_LOF) != -1]
        print(f"Total outlier remove by LOF:", self.Outlier_LOF.shape[0])
        #Test
        LOF = LocalOutlierFactor(n_neighbors = self.n_neighbors, novelty = True)
        LOF.fit(self.data_train_LOF)
        self.Data_test_LOF = self.data_test_LOF[LOF.predict(self.data_test_LOF) != -1]


