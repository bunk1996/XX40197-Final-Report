import numpy as np
from numpy import pi
from scipy import fft

def downconvert(fc,msk,fs,BW,N):
    """
    Function to downconvert and demodulate MSK signal
    IN
    fc: centre freq Hz
    msk: modulated time series
    fs: sampling frequency of time series
    BW: bandwidth
    N: length signal in samples
    OUT
    inst_phase: Instantaneous phase angle
    inst_freq: instantaneous frequemcy found by calculating
    first diff of phase
    """
    
    t = np.r_[0:N]/fs
    Fc = np.exp(2*pi*fc*t*1j)
    msk_dc = np.multiply(Fc, msk)
    dc_f = fft.fft(msk_dc)
    xf = fft.fftfreq(len(msk),1/fs)
    cutoff = 2*BW
    filt = np.where((xf <= -cutoff) | (xf >= cutoff)) 
    dc_f[filt] = 0

    dc = fft.ifft(dc_f)
    y = dc.real
    z = dc.imag

    inst_phase = -np.arctan2(z,y) + pi/4
    inst_phase = np.unwrap(inst_phase) - pi/4
    inst_freq = np.diff(inst_phase)*fs/(2*pi)
    return inst_phase, inst_freq, dc