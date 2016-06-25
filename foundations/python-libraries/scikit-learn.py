import numpy as np
import sklearn as sk
from sklearn import preprocessing

data = np.genfromtxt('sample.data',
                     delimiter=',',
                     dtype='f8,f8,f8,f8',
                     names='a,b,c,d')

# this method of import creates a flexible type which
# does not allow `mean.()` to be called
X = np.array([data[x] for x in data.dtype.names])
mean = X.mean(axis=0)
stdev = X.std(axis=0)


# The scale method from sklearn.preprocessing subtracts the mean and divides
# by the standard deviation. Use preprocessing.scale to scale the data.
# The scaled data should be stored in the scaledData variable.
scaledData = preprocessing.scale(X)

scaledData.mean(axis=0)
scaledData.std(axis=0)


scaler = preprocessing.StandardScaler().fit(X)
