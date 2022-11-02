import numpy as np
import nprirgen.funcs as funcs


# Generate RIRs.
def genrir(soundVelocity, fs, receiverPositions, sourcePosition, roomMeasures, betaCoeffs, orientation,
           isHighPassFilter, nOrder, nSamples, micType):
    if nOrder != -1:
        raise ValueError('nOrder must be -1.')

    if micType != 'omnidirectional':
        raise ValueError('micType must be omnidirectional')

    nMicrophones = len(receiverPositions)

    # Check if the reflection order is valid.
    if nOrder < -1:
        raise ValueError('Reflection order must be an integer >= -1.')

    W = 2 * np.pi * 100 / fs  # cut-off freq (100 Hz)
    R1 = np.exp(-W)
    B1 = 2 * R1 * np.cos(W)
    B2 = -R1 * R1
    A1 = -(1 + R1)

    # tmp vars related to RIR generation
    Fc = 1  # The cut-off frequency equals fs/2 - Fc is the normalized cut-off frequency.
    Tw = 2 * funcs.symceil(0.004 * fs)  # The width of the low-pass FIR equals 8 ms
    cTs = soundVelocity / fs;

    Rp_plus_Rm = np.zeros(3)
    refl = np.zeros(3)
    LPI = np.zeros(Tw)

    imp = np.zeros((nMicrophones, nSamples))

    s = sourcePosition / cTs
    L = roomMeasures / cTs

    n = np.amax(np.ceil(nSamples / (2 * L)).astype(int))

    Rm0 = np.arange(-n, n + 1) * 2 * L[0]
    Rm1 = np.arange(-n, n + 1) * 2 * L[1]
    Rm2 = np.arange(-n, n + 1) * 2 * L[2]

    Rp_plus_Rm0 = np.stack((Rm0 + s[0], Rm0 - s[0]), axis=1)
    Rp_plus_Rm1 = np.stack((Rm1 + s[1], Rm1 - s[1]), axis=1)
    Rp_plus_Rm2 = np.stack((Rm2 + s[2], Rm2 - s[2]), axis=1)

    refl0_1 = np.power(betaCoeffs[0], np.abs(np.arange(-n, n + 1)))
    refl0_0 = np.power(betaCoeffs[1], np.abs(np.arange(-n - 1, n + 1)))
    refl0 = np.stack((np.multiply(refl0_0[1:], refl0_1), np.multiply(refl0_0[:-1], refl0_1)), axis=1)

    refl1_1 = np.power(betaCoeffs[2], np.abs(np.arange(-n, n + 1)))
    refl1_0 = np.power(betaCoeffs[3], np.abs(np.arange(-n - 1, n + 1)))
    refl1 = np.stack((np.multiply(refl1_0[1:], refl1_1), np.multiply(refl1_0[:-1], refl1_1)), axis=1)

    refl2_1 = np.power(betaCoeffs[4], np.abs(np.arange(-n, n + 1)))
    refl2_0 = np.power(betaCoeffs[5], np.abs(np.arange(-n - 1, n + 1)))
    refl2 = np.stack((np.multiply(refl2_0[1:], refl2_1), np.multiply(refl2_0[:-1], refl2_1)), axis=1)

    refl = np.einsum('ij,kl,mn->ijklmn', refl0, refl1, refl2)

    inds = np.arange(1, Tw + 1)

    for idxMicrophone in range(nMicrophones):
        r = receiverPositions[idxMicrophone] / cTs

        tmp = np.add.outer((Rp_plus_Rm0 - r[0]) ** 2, (Rp_plus_Rm1 - r[1]) ** 2)
        dist = np.sqrt(np.add.outer(tmp, (Rp_plus_Rm2 - r[2]) ** 2))
        fdist = np.floor(dist)
        ddist = dist - fdist

        inds_minus_ddist = np.subtract.outer(inds, ddist)
        LPI = Fc * 0.5 * np.multiply(1 - np.cos(2 * np.pi * inds_minus_ddist / Tw),
                                     np.sinc(Fc * (inds_minus_ddist - Tw / 2)))

        startPositions = (fdist - Tw / 2 + 1).astype(int)
        src_b = np.maximum(np.maximum(startPositions, 0) - startPositions, 0)
        src_e = np.maximum(np.minimum(startPositions + Tw, nSamples) - startPositions, 0)
        dest_b = src_b + startPositions
        dest_e = src_e + startPositions

        gain = np.divide(refl, dist) / (4 * np.pi * cTs)

        for mx in range(2 * n + 1):
            for my in range(2 * n + 1):
                for mz in range(2 * n + 1):
                    for q in range(0, 2):
                        for j in range(0, 2):
                            for k in range(0, 2):
                                if fdist[mx, q, my, j, mz, k] < nSamples:
                                    imp[idxMicrophone, dest_b[mx, q, my, j, mz, k]: dest_e[mx, q, my, j, mz, k]] += \
                                    gain[mx, q, my, j, mz, k] * LPI[
                                                                src_b[mx, q, my, j, mz, k]: src_e[mx, q, my, j, mz, k],
                                                                mx, q, my, j, mz, k]

            # 'Original' high-pass filter as proposed by Allen and Berkley.
        if isHighPassFilter:
            Y = np.zeros(3)

            for idx in range(nSamples):
                X0 = imp[idxMicrophone, idx]
                Y[2] = Y[1]
                Y[1] = Y[0]
                Y[0] = B1 * Y[1] + B2 * Y[2] + X0
                imp[idxMicrophone, idx] = Y[0] + A1 * Y[1] + R1 * Y[2]

    return imp