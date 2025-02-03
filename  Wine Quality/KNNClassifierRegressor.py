import numpy as np


class KNNRegressor:
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors


    def fit(self,X,Y):
        self.X_train = X
        self.y_train = Y

    #Euclide
    def distance(self,x1,x2):
        return np.sqrt(np.sum((x1**2-x2)**2))

    def predict(self,X):
            y_pred=[self.predict(x) for x in X]
            return np.array(y_pred)
    def _predict(self,x):
        distances=[self.disctance(x,x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_values = [self.y_train[i] for i in k_indices]
        return np.mean(k_nearest_values)
        