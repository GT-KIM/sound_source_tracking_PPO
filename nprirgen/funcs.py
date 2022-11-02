import numpy as np


# Symmetric ceiling function
def symceil(x):
    return np.round(x + np.sign(x) * (0.5 - np.finfo(float).eps)).astype(int)


_rho = {'bidirectional': 0,
        'hypercardioid': 0.25,
        'cardioid': 0.5,
        'subcardioid': 0.75,
        'omnidirectional': 1}


def sim_microphone(x, y, z, angle, mtype):
    vartheta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    varphi = np.arctan2(y, x)

    gain = np.sin(np.pi / 2 - angle[1]) * np.sin(vartheta) * np.cos(angle[0] - varphi) + np.cos(
        np.pi / 2 - angle[1]) * np.cos(vartheta)
    gain = _rho[mtype] + (1 - _rho[mtype]) * gain

    return gain