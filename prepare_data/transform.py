import numpy as np

def tfdataset2array(dataset):
    xdata=[]
    ydata=[]
    for x, y in dataset.as_numpy_iterator():
        xdata.append(x)
        ydata.append(y)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    ydata.reshape([-1,1])
    return xdata, ydata