import numpy as np
from scipy import stats
from tqdm import tqdm
from medianExtraction import median_frequency

def zeroXing(instf):
    """
    Function to find all zero crossings in a time series, using a method based of changed in sign.
    instf: Instantaneous frequency of baseband - array like
    xing: All zero crossings in the signal - array like
    """
    sign = np.sign(instf)
    xing = np.array([])

    for s in tqdm(range(0,len(sign)-2)):
        if sign[s] == sign[s+1]:
            pass
        elif sign[s] != sign[s+1]:
            xing = np.append(xing,s)
    return np.sort(xing.astype(int))


def basicExctract(instf,instp, Ns):
    """
    Basic system of extracting bits from instantaneous frequency 
    instf: instantaneous frequency
    Ns: No. of samples per bit
    xing: everytime the instantantaneos frequency crosses zero

    The method simply determines whether the crossing is up or down and calculates how many bits are in that period
    """
    # initialise
    xing = zeroXing(instf)
    xing = np.sort(xing)
    # xing = remove_multiples(instf, Ns)
    L = len(instf)
    l = len(xing)
    bits = np.array([])
    symdex = np.array([0])
    prev = 0
    totBits = np.round(L/Ns)
    # middle bits
    for x in tqdm(range(0,l)):
        i = int(xing[x])
        if i < l:
            nxt = int(xing[x+1])
        else:
            nxt = int(instp[-1])
        fdiff = nxt - i
        diff = i - prev 
        grad = np.divide((instp[i]-instp[prev]),diff)
        grad *= 1e6/(2*np.pi)
        nxtgrad = np.divide((instp[nxt]-instp[i]), fdiff)
        nxtgrad *= 1e6/(2*np.pi)
        # grad = (instf[i+1] - instf[i-1])/2
        nBits = diff / Ns
        
        # Conditional to 
        if nBits > 0.8:
            symdex = np.append(symdex,i)
            prev = int(xing[x])
            # check right no. of symbols detected
            acBits = len(bits)
            corBits = np.round(i/Ns)
            bitDiff = int(corBits-acBits)
            
            # Check to ensure right no.of bitsx
            if bitDiff != 0:
                if bitDiff > 0:
                    nBits = bitDiff
                else:
                    nBits = 0

            
        if np.sign(grad) != np.sign(nxtgrad):
            if grad < 0:
                # +ve crossing ie 0 to 1
                symbol = np.zeros(int(nBits))
                bits = np.append(bits,symbol)
            elif grad > 0:
                symbol = np.ones(int(nBits))
                bits = np.append(bits,symbol)
        elif np.sign(grad) == np.sign(nxtgrad):
            if grad < 0:
                # +ve crossing ie 0 to 1
                symbol = np.zeros(int(nBits))
                bits = np.append(bits,symbol)
            elif grad > 0:
                symbol = np.ones(int(nBits))
                bits = np.append(bits,symbol)

    # last bits
    diff = len(instp)-1 - xing[x]
    grad = np.divide((instp[-1]-instp[x]),diff)
    grad *= 1e6/(2*np.pi)
    
    nBits = diff/Ns
    acBits = len(bits)
    corBits = np.round(L/Ns)
    nBits = corBits-acBits
    
    if grad < 0:
        # +ve crossing ie 0 to 1
        symbol = np.zeros(int(nBits))
        bits = np.append(bits,symbol)
    elif grad > 0:
        symbol = np.ones(int(nBits))
        bits = np.append(bits,symbol)


    return bits, symdex



    