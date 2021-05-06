

# Generate Minimum Shift Keyed signal
import numpy as np
from scipy import fft
from numpy import pi
from boxcar import boxcar
import noise
import bitstring as bs
from tqdm import tqdm
# set up the message and convert to binary list
class mskSim:
    def __init__(self,fc,fs,BW,msg):
        self.fs = fs
        self.fc = fc
        self.bw = BW
        self.T = 1/BW
        self.Ns = fs/BW
        self.msg = msg

    def genMsg(self):
        msg = bs.BitArray(bytes=bytes(self.msg, 'utf-8'))
        msgl = [char for char in msg]
        self.L = len(msgl)
        return msgl
    

    def genSig(self,input,msg):
        if input == 'Text':
            msgl = self.genMsg()
            msgl = np.array(msgl)*2-1
        elif input == 'Bits':
            msgl = msg
            self.L = len(msgl)
            pass

        a = np.array([1])
        b = np.array([1])

        for i in tqdm(range(0,self.L-2)):
            if i%2 == 0:
                if msgl[i] == msgl[i+1]:
                    a = np.append(a, a[int(i/2)])
                elif msgl[i] != msgl[i+1]:
                    a = np.append(a, a[int(i/2)]*-1)
            else:
                if msgl[i] == msgl[i+1]:
                    b = np.append(b, b[int(np.floor(i/2))])
                elif msgl[i] != msgl[i+1]:
                    b = np.append(b, b[int(np.floor(i/2))]*-1)

        a = np.tile(a, (int(2*self.Ns),1))
        a = a.transpose()
        a = a.ravel()
        a = np.append(a, a[int(len(a)-self.Ns):])

        b = np.tile(b, (int(2*self.Ns),1))
        b = b.transpose()
        b = b.ravel()
        
        self.N = self.L*self.Ns
        self.t = np.r_[0:self.N]/self.fs
        t_i = np.r_[-self.Ns:self.N]/self.fs
        
        I = a*np.cos((pi*t_i)/(2*self.T))
        I = I[int(self.Ns):]
        Q = b*np.sin((pi*self.t)/(2*self.T))
        
        alpha = 2*pi*self.fc
        self.mskcos = I*np.cos(alpha*self.t)
        self.msksin = Q*np.sin(alpha*self.t)
        self.msk = self.mskcos + self.msksin
        return self.msk

        
        
    