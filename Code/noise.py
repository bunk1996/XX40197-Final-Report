import numpy as np
from numpy import pi
from boxcar import boxcar
import random
import scipy.stats as stats
from tqdm import tqdm

class rand_noise():
    def __init__(self,data, fs, flow,fupp,t):
        self.data = data
        self.fs = fs
        self.flow = flow
        self.fupp = fupp
        self.t = t

    def APD(self):
        """
        Generates APD from desired data set
        yf: filtered signal
        hist: histogram representing APD
        """
        yf = boxcar(self.data,self.fs,self.flow,self.fupp)
        ph = np.arctan2(yf.imag, yf.real)
        self.sigma = np.std(abs(yf))/2
        self.mu = np.mean(abs(yf))
        self.yf = abs(yf)
        L = len(yf)
        hist = np.histogram(abs(self.yf), bins = 'auto')
        hist_scaled = np.array(hist[0] / np.sum(hist[0]))
        hist_scaled = np.insert(hist_scaled, 0, 0)
        self.hist = np.array([np.asarray(hist[1]), hist_scaled.transpose()])
        

    def rand_noise(self, simData,tmax, AWGN):
        """
        Method to generate and append added noise to simulated signal
        simData: Simulated Waveform
        y : noise corrupted simulation
        """
        self.APD()
        cdf = np.cumsum(self.hist[1])
        A = abs(self.yf)
        L = len(simData) 
        lbound = int(len(simData)/4)
        ubound = int(3*lbound)
        
        if AWGN == True:
            signal = self.sigma * np.random.randn(ubound-lbound) + self.mu
            simData[lbound:ubound] = np.add(simData[lbound:ubound],signal)
        else:
            pass
        
        n = int(20)
        burst = random.randint(100,20000)
        x = self.hist[0]
        for count in tqdm(range(0,n)):
            j = random.randint(lbound,ubound)
            lim = j+burst
            ph = random.random() * 2 * pi
            while j < int(lim):
                if burst < 5000:
                    k = 10
                else:
                    k = 5
                p = random.random()
                i = np.where(cdf >= p)[0][0]
                spike = k*x[i]
                sign = random.random()
                spike = spike*(np.cos(ph) + np.sin(ph)*1j)
                simData[j] = simData[j] + spike
                ph = ph + (2*pi/self.fs)
                j+=1
        y = simData
        return y