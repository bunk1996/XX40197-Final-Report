from mskSim import mskSim

from downconvert import downconvert

import matplotlib.pyplot as plt

from scipy import fft
import numpy as np
from numpy import pi
import noise
import scipy.io as scio

import bitExtract
from snr import SNR,SNR_per_bit,pwr,noise_estimate
    #Initialise function
    # Define Simulation parameters
from phase_estimate import phase_estimate

def tile(Ns, y):
    x = np.tile(y,(int(Ns),1))*2-1
    x = x.transpose()
    return x.ravel()

fs = 1e6
fc = 24e3
BW = 200
flow = -3*BW
fupp = 3*BW
pb = [flow,fupp]


msg = 'quick brown fox jumps over the lazy dog'

Nbits = 400
np.random.seed(seed=1)
bits = np.random.randn(Nbits,1) > 0

bits = bits * 2-1
# M = np.tile(bits,(1,Ns))

sim = mskSim(fc,fs,BW,msg)
# dm = demod(fs,fc,pb,BW)

msk = sim.genSig('Bits',bits)
msk /= 250
import gaussian
# msk = gaussian.gauss_filter(msk)
np.save('msk.npy',msk)

xf = fft.fftfreq(len(msk),1/fs)
inst_phase, inst_freq,dc = downconvert(fc,msk,fs,BW,len(msk))
t = sim.t

bits = bitExtract.basicExctract(inst_freq,inst_phase, sim.Ns)

mat = scio.loadmat('/Users/benmonk/Documents/Uni/XX40197/Code/matlab/y1mhz.mat')
data = mat['y']

tmax = np.max(t)

randNoise = noise.rand_noise(data, fs, flow, fupp,tmax)
randNoise.APD()
# plt.plot(msk,linestyle ='--')
k = 1

mskf = fft.fft(msk)
mski = fft.ifft(mskf)
noiseSignal = randNoise.rand_noise(mski,t,False,k)
noiseF = fft.fft(noiseSignal)
noiseSig = fft.ifft(noiseF)
noiseSignal = noiseSig.real


# pwr_noise = pwr(noisyS)
inst_phase_noise, inst_freq_noise,dcn = downconvert(fc,noiseSignal,fs,BW,len(noiseSignal))
np.save('ipn.npy',inst_phase_noise)
noiseBits = bitExtract.basicExctract(inst_freq_noise,inst_phase_noise, sim.Ns)

# plt.legend(loc=1)
# plt.show()

snr = SNR(noiseSignal,fs)

ones = np.where(bits[0] == 1)[0]
zeros = np.where(bits[0] == 0)[0]
print(ones.size, zeros.size)
ones = np.where(noiseBits[0] == 1)[0]
zeros = np.where(noiseBits[0] == 0)[0]
print(ones.size, zeros.size)

bitLocOg = bits[1]
bitlocNoise = noiseBits[1]
print(bitlocNoise[:25])
# indDiff = bitLocOg - bitlocNoise

diff = np.array(bits[0]) - np.array(noiseBits[0])
error = np.where(diff != 0)[0]
error = np.multiply(error, sim.Ns)

print("SNR: ", snr)

print('BER:', np.log((error.size/sim.L)))
# print(np.array(bits[0]) - np.array(noiseBits[0]))

ogMsg = tile(sim.Ns,bits[0])
noiseMsg = tile(sim.Ns, noiseBits[0])

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
    
    sym = y[(i+1):(len(y)-1)]
    fbar = np.append(fbar, median_correction(sym))    
    return fbar

msk = np.load('msk.npy', allow_pickle=True)
fbar = median_frequency(msk,bitLocOg)

b = median_frequency(inst_phase,bitLocOg)
a = median_frequency(inst_phase_noise,bitlocNoise)




binsFreq = np.histogram_bin_edges(((inst_phase/pi)%0.5),bins = 9)
binsMedian = np.histogram(abs(med_freq), bins = 10)
maxind = np.argmax(binsMedian[0])
binsMedian = binsMedian[1]
thrF = np.array([binsFreq[1],binsFreq[-2]])
thrMed = np.array([binsMedian[maxind],binsMedian[maxind+1]])
phChg = inst_phase_noise[bitlocNoise]
phChg = (phChg/pi)%0.5

med = np.where((abs(a) < thrMed[0]) | (abs(a) > thrMed[1]))[0]
ph = np.where((phChg > thrF[0]) & (phChg < thrF[1]))[0]
com = np.array([])
for p in ph:
    c = np.where(med == p)[0]
    com = np.append(com, med[c])

com = com.astype(int)
newPhase = phase_estimate(inst_phase_noise,sim.Ns,bitlocNoise[com])
newFreq = np.diff(newPhase)*fs/(2*pi)

newBits = bitExtract.basicExctract(newFreq,newPhase,sim.Ns)
newMsg = tile(sim.Ns, newBits[0])
diff1 = np.array(bits[0]) - np.array(newBits[0])
error1 = np.where(diff1 != 0)[0]
error1 = np.multiply(error1, sim.Ns)
print('BER1:', np.log((error1.size/sim.L)))
mskf = fft.fft(msk)
mski = fft.ifft(mskf)
