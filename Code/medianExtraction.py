import numpy as np
from numpy import pi
def median_correction(y):
    Q1 = np.percentile(y, 25, interpolation = 'midpoint')
    # Third quartile (Q3)
    Q3 = np.percentile(y, 75, interpolation = 'midpoint')
    i = np.where((y < Q1) | (y > Q3))[0]
    iqr = y[i]
    fbar = np.median(np.diff(y)*(1e6/(2*pi)))
    return fbar

def median_frequency(y,ind):
    ind = ind.astype(int)
    fbar = np.array([])
    for i in range(0,len(ind)-1):
        sym = y[ind[i]:ind[i+1]]
        fbar = np.append(fbar, median_correction(sym))
    
    # sym = y[(i+1):(len(y)-1)]
    # fbar = np.append(fbar, median_correction(sym))    
    return fbar