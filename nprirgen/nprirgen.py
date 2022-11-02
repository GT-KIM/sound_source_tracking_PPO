import numpy as np

# Reference https://github.com/ty274/rir-generator
def np_generateRir(roomMeasures, sourcePosition, receiverPositions, *, reverbTime=None, betaCoeffs=None,
                   soundVelocity=340, fs=16000, orientation=[.0, .0], isHighPassFilter=True, nDim=3, nOrder=-1,
                   nSamples=-1, micType='omnidirectional'):
    """ NumPy implementation of the RIR generator developed by Marvin182 (https://github.com/Marvin182/rir-generator).

    Room Impulse Response Generator

    Computes the response of an acoustic source to one or more microphones in a reverberant room using the image method.
    Author : ty274
    Copyright (C) 2017- Takuya Yoshioka

    Args:
    	roomMeasures: 1d array-like with 3 elements specifying the room dimensions (x,y,z) in m
    	sourcePosition: 1d array0like with 3 elements specifying the (x,y,z) coordinates of the source in m
    	receiverPositions: 1d (of length 3) or 2d (or shape Mx3) array specifying the (x,y,z) coordinates of the receiver(s) in m
    	reverbTime (float): reverberation time (T_60) in seconds
    	betaCoeffs: 1d vector with 6 elements specifying the reflection coefficients [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2]
        soundVelocity: sound velocity in m/s
    	fs: sampling frequency in Hz
    	orientation: array-like specifying direction in which the microphones are pointed, represented with azimuth and elevation angles (in radians), default is [0 0]
    	isHighPassFilter: use 'False' to disable high-pass filter, the high-pass filter is enabled by default
    	nDim: room dimension (2 or 3), default is 3
    	nOrder: reflection order, default is -1, i.e. maximum order
    	nSample: number of samples to calculate, default is T_60*fs
    	micType: [omnidirectional, subcardioid, cardioid, hypercardioid, bidirectional], default is omnidirectional
    Return:
        M x nsample array containing the calculated room impulse response(s)
    """

    # Convert the inputs to ndarrays
    #roomMeasures = np.array(roomMeasures)
    #sourcePosition = np.array(sourcePosition)
    #receiverPositions = np.array(receiverPositions)

    if receiverPositions.ndim == 1:
        receiverPositions = receiverPositions[np.newaxis, :]

    # Only one of reverbTime and betaCoeffs must be given.
    if not (reverbTime is None) != (betaCoeffs is None):
        raise ValueError('You must provide either reverbTime or betaCoeffs.')

    # Calculate the reflection coefficients from the T60.
    if betaCoeffs is None:
        V = np.prod(roomMeasures)
        S = 2 * (roomMeasures[0] * roomMeasures[2] + roomMeasures[1] * roomMeasures[2] + roomMeasures[0] * roomMeasures[
            1])

        if reverbTime > 0:
            alpha = 24 * V * np.log(10) / (soundVelocity * S * reverbTime)
            betaCoeffs = np.ones(6) * np.sqrt(1 - alpha)
        elif reverbTime == 0:
            betaCoeffs = np.zeros(6)
        else:
            raise ValueError('T60 must be >= 0.')

    # Calculate the T60 from the reflection coefficients.
    if reverbTime is None:
        V = np.prod(roomMeasures)
        alpha = ((1 - betaCoeffs[0] ** 2) + (1 - betaCoeffs[1] ** 2)) * roomMeasures[1] * roomMeasures[2] + (
                    (1 - betaCoeffs[2] ** 2) + (1 - betaCoeffs[3] ** 2)) * roomMeasures[0] * roomMeasures[2] + (
                            (1 - betaCoeffs[4] ** 2) + (1 - betaCoeffs[5] ** 2)) * roomMeasures[0] * roomMeasures[1]
        reverbTime = 24 * np.log(10) * V / (soundVelocity * alpha)
        if reverbTime < 0.128:
            reverbTime = 0.128

    # Determine the number of samples to generate.
    if nSamples == -1:
        nSamples = int(reverbTime * fs)

    # Disable elevation when nDim=2.
    if nDim == 2:
        betaCoeffs[4] = 0
        betaCoeffs[5] = 0

    # Now, generate RIRs.
    import nprirgen.gen as gen
    h = gen.genrir(soundVelocity, fs, receiverPositions, sourcePosition, roomMeasures, betaCoeffs, orientation,
                   isHighPassFilter, nOrder, nSamples, micType)

    return np.squeeze(h), reverbTime, betaCoeffs