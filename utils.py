import numpy as np
import math
import torch
import scipy.signal
def AWGN(inputs, target_snr) :
    watts = inputs ** 2
    sig_avg_watts = np.mean(watts) + 1e-10
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg_watts), size=len(inputs))
    return inputs + noise

def SNR(noisy, noise) :
    dt = 0.001
    Signal = np.sum(np.abs(np.fft.fft(noisy)*dt)**2) / len(np.fft.fft(noisy))
    Noise = np.sum(np.abs(np.fft.fft(noise)*dt)**2) / len(np.fft.fft(noise))
    print(10 * np.log10(Signal/Noise))

def polar2cart(r, theta) :
    rad = np.deg2rad(theta)
    x = r * np.sin(rad)
    y = r * np.cos(rad)
    return x, y

def cart2polar(x, y) :
    r = math.sqrt(pow(x,2) + pow(y,2))
    theta = math.atan2(y, x)

    return r, theta

def moving_avg(x, n=10) :
    return np.convolve(x, np.ones(n), 'same') / n

def discount(x, gamma, terminal_array=None) :
    if terminal_array is None :
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else :
        y, adv = 0, []
        terminals_reversed = terminal_array[1:][::-1]
        for step, dt in enumerate(reversed(x)) :
            y = dt + gamma * y * (1 - terminals_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]

class RunningStats(object) :
    def __init__(self, epsilon=1e-4, shape=()) :
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x) :
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) :
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count

def lstm_state_combine(state) :
    return np.reshape([s[0] for s in state], (len(state), -1)), np.reshape([s[1] for s in state], (len(state), -1))