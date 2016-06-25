import numpy as np

data = np.genfromtxt('sample2.csv', delimiter=',', dtype='f8,i8,S15,S15', names=True)

print data
print data.dtype.names

data['Field0'].mean()
data['Field0'].max()
data['Field0'].argmax()
data['Field0'].std()


'''There are times when we would like to ignore the strings in our CSV and just work with the numbers. In order to skip the header, skiprows, and only load the columns with numbers, usecols, enter this line into the console:'''
data2 = np.loadtxt('sample2.csv', delimiter=',', usecols=(0,1), skiprows=1)
data2.T
