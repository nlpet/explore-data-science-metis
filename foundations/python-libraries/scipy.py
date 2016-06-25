import scipy as sp
import numpy as np
from scipy import stats

# help
help(open)
sp.info("mean")
sp.info(sp.mean)


data = np.genfromtxt(
    'iris.data',
    delimiter=',',
    dtype='f8,f8,f8,f8,S15',
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type_string']
)

sizeofdata, (minval,maxval), mean, variance, skew, kurtosis = stats.describe(data['petal_width'])
sp.median(data['petal_length'])
