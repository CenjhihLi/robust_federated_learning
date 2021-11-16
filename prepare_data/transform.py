import numpy as np

def tfdataset2array(dataset):
    xdata=[]
    ydata=[]
    for x, y in dataset.as_numpy_iterator():
        xdata.append(x)
        ydata.append(y)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    return xdata, ydata

def tfgenerator2array(generator):
    xdata=[]
    ydata=[]
    for x, y in generator:
        xdata.append(x)
        ydata.append(y)
    xdata = np.concatenate(xdata, axis = 0)
    ydata = np.concatenate(ydata, axis = 0)
    return xdata, ydata