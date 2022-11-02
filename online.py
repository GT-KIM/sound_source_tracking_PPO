from utils import *
from librosa import stft
#import signal
#from socket import *

N = 1024

def array_spectrogram(frames, N = 1024, fs=16000) :
    spec = list()
    for i in range(len(frames)) :
        current_spec = stft(frames[i], n_fft=N, hop_length=int(N / 2), win_length=N, window='bohman')
        spec.append(current_spec)
    return spec

def srp_phat(frames, W, nMic=8) :#, ax1, ax2) :

    Q = W.shape[0]
    num_of_microphone = 8
    spec = array_spectrogram(frames)
    Y = np.zeros(Q, dtype=np.complex_)

    for iter_Q in range(Q):
        iter_mic = 0
        for iter_mic1 in range(num_of_microphone) :
            for iter_mic2 in range(iter_mic1 + 1, num_of_microphone) :
                X1 = spec[iter_mic1][:,1]
                X2 = spec[iter_mic2][:,1]
                X = np.multiply(X1, X2.conjugate())
                X = np.divide(X, np.abs(X) + 1e-10)
                Y[iter_Q] += np.dot(W[iter_Q, iter_mic, :],X)
                iter_mic += 1
    Y = np.real(Y)
    return Y

"""
def stereo2mono(stereo) :
    buf = np.fromstring(stereo, dtype='int16')
    frame1 = buf[0::2]
    frame2 = buf[1::2]
    byte1 = frame1.tostring()
    byte2 = frame2.tostring()

    return byte1, byte2

def byte2array(frames1, frames2, frames3, frames4, fs=16000, record_time=3) :
    raw_array = np.zeros((fs * record_time, 4))
    raw_array[:, 0] = np.fromstring(frames1, np.int16)
    raw_array[:, 1] = np.fromstring(frames2, np.int16)
    raw_array[:, 2] = np.fromstring(frames3, np.int16)
    raw_array[:, 3] = np.fromstring(frames4, np.int16)

    return raw_array


top4 = Y.argsort()[-4:][::-1]
#bestY = np.zeros(Y.shape)
#bestY[top3] = Y[top3]
polar = Q_polar[:, top4]
for i in range(4) :
    polar[1,i] = rad2deg(polar[1,i])
#print(polar)
angle = np.average(polar[1,:])

if abs(np.max(polar[1,:]) - np.min(polar[1,:])) > 30 or 85 <= angle <= 95 :
    return int(angle), False
else :
    return int(angle), True

def signal_handler(sig, frame) :
    print("Exit")
    sys.exit(0)

def main() :
    #fig = plt.figure()
    #plt.ion()
    #ax1 = plt.subplot(2, 1, 1)
    #ax2 = plt.subplot(2, 1, 2)
    offline = np.load("./W_fastest.npz")
    W = offline['W']
    Q_array = offline['Q']
    Q_polar = offline['Q_polar']
    signal.signal(signal.SIGINT, signal_handler)

    #connectionSock = socket(AF_INET, SOCK_STREAM)
    #connectionSock.connect(('127.0.0.1', 1234))
    #_ = input("Press Enter to Start")
    frame1, fs = librosa.load("C:\\Users\GTKim\Documents\MATLAB\RIR\\RIR-Generator-master\wav1.wav")
    frame2, fs = librosa.load("C:\\Users\GTKim\Documents\MATLAB\RIR\\RIR-Generator-master\wav2.wav")
    frame3, fs = librosa.load("C:\\Users\GTKim\Documents\MATLAB\RIR\\RIR-Generator-master\wav3.wav")
    frame4, fs = librosa.load("C:\\Users\GTKim\Documents\MATLAB\RIR\\RIR-Generator-master\wav4.wav")
    plt.subplot(411)
    plt.plot(frame1)
    plt.subplot(412)
    plt.plot(frame2)
    plt.subplot(413)
    plt.plot(frame3)
    plt.subplot(414)
    plt.plot(frame4)
    plt.show()
    angle, flag = srp_phat(frame1, frame2,frame3, frame4, W=W, Q_polar=Q_polar)#, ax1, ax2)
    print(angle)
if __name__ == '__main__':
    main()
"""