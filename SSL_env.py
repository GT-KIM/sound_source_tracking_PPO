import numpy as np
import nprirgen.nprirgen as RG
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
from offline_v2 import *
from online import *
from scipy import special as sp
from scipy import stats as st
from utils import *
#from bresenham import bresenham

def get_mic_pos(origin, r=0.3, nMic = 8, Offline=False) :
    theta = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    x, y = polar2cart(r, theta)
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    z = np.ones((nMic,1))
    mic = np.concatenate((x,y,z), axis=1)
    if Offline :
        # ax = plt.subplot(111)
        # ax.scatter(mic[:,0],mic[:,1], cmap='hsv')
        # plt.show()
        # plt.close()
        return mic
    else :
        return mic + origin[np.newaxis, :]#np.sum((mic, origin[np.newaxis, :]))

class SSL_env :
    def __init__(self, nsamples=1024, nMic=8) :
        self.n_obstacles = 2
        self.T = 26                         # Temperature (C)
        self.c = 331.3 + 0.606 * self.T     # Sound velocity (m/s)
        self.fs = 16000                     # Sample frequency (samples/s)
        self.rt = 2.0                       # Reverberation time (s)
        self.noise_dB = 10
        self.nSamples = nsamples                   # Number of samples
        self.nMic = nMic                    # Number of Microphones
        self.flag_offline = False           # True : do offline process, False : load exist W coefficient
        self.flag_shape = 'rect'            # 'polar', 'rect' : srp coordinate
        self.Rdim = np.array([20, 20, 5])   # Room dimensions [x y z] (m)
        self.Sdim = np.array([4, 4, 1], dtype=np.float32)   # Array of Source position [x y z] (m)
        self.roomgrid = np.zeros((self.Rdim[0] * 10, self.Rdim[1] * 10))  # room grid
        self.obstacle_room = np.zeros((self.Rdim[0] * 10, self.Rdim[1] * 10))
        self.initial_pos = np.array([3.5, 3.5, 1], dtype=np.float32)      # agent init pos
        self.pos = self.initial_pos.copy()
        self.threshold = 1000
        self.distance_old = 0
        self.reward_clip = False

        self.obstacles = self.generate_obstacles(self.n_obstacles)

        # srp-phat offline process
        if self.flag_offline:
            mic = get_mic_pos(self.pos, Offline=True)
            if self.flag_shape == 'polar':
                self.Q_array, self.Q_polar, self.W = offline_SRP(mic)
            if self.flag_shape == 'rect':
                self.Q_array, self.W = offline_SRP_rect(mic)
        else:
            if self.flag_shape == 'polar':
                d = np.load("./W_polar_v2.npz")
                self.Q_array = d['Q']
                self.Q_polar = d['Q_polar']
                self.W = d['W']
            if self.flag_shape == 'rect':
                d = np.load("./W_rect_v2.npz")
                self.Q_array = d['Q']
                self.W = d['W']

        plt.rcParams["figure.figsize"] =(8,8)

    def generate_obstacles(self, num_obstacles = 2) :
        self.obstacle_room = np.zeros((self.Rdim[0] * 10, self.Rdim[1] * 10))

        obstacles = list()
        self.obstacle_room[:19, :] = 1
        self.obstacle_room[181:, :] = 1
        self.obstacle_room[:,:19] = 1
        self.obstacle_room[:, 181:] = 1

        for i in range(num_obstacles) :
            obstacle_x = np.random.rand(1) * 0.8 + 0.1
            obstacle_y = np.random.rand(1) * 0.8 + 0.1
            size_x = np.random.rand(1) * 0.3 + 0.1
            size_y = np.random.rand(1) * 0.3 + 0.1

            position = [0, 0, 0, 0] # x_low, x_high, y_low, y_high
            position[0] = int(self.obstacle_room.shape[0] * obstacle_x)
            position[1] = int(self.obstacle_room.shape[0] * (obstacle_x + size_x))
            if position[1] > self.obstacle_room.shape[0] :
                position[1] = self.obstacle_room.shape[0] - 1
            position[2] = int(self.obstacle_room.shape[1] * obstacle_y)
            position[3] = int(self.obstacle_room.shape[1] * (obstacle_y + size_y))
            if position[3] > self.obstacle_room.shape[1] :
                position[3] = self.obstacle_room.shape[1] - 1

            self.obstacle_room[position[0] : position[1], position[2] : position[3]] = 1
            obstacles.append(position)
        return obstacles



    def acting(self, action) :
        pos_old = self.pos.copy()
        if action == 0:
            self.pos[0] -= 0.5
            self.pos[1] -= 0.5
        if action == 1 :
            self.pos[0] -= 0.5
        if action == 2 :
            self.pos[0] -= 0.5
            self.pos[1] += 0.5
        if action == 3 :
            self.pos[1] -= 0.5
        if action == 5 :
            self.pos[1] += 0.5
        if action == 6 :
            self.pos[0] += 0.5
            self.pos[1] -= 0.5
        if action == 7 :
            self.pos[0] += 0.5
        if action == 8 :
            self.pos[0] += 0.5
            self.pos[1] += 0.5

        act_reward = 0
        if self.pos[0] < 2 or self.pos[0] > 18 or self.pos[1] < 2 or self.pos[1] > 18 :
            act_reward = -10
            terminal = False
            self.pos = pos_old
        elif self.obstacle_room[int(self.pos[0] * 10), int(self.pos[1] * 10)] == 1 :
            act_reward = -10
            terminal = False
            self.pos = pos_old
        else :
            terminal = False

        if self.reward_clip :
            return act_reward, terminal
        else :
            return act_reward, terminal

    def reward_action(self, action, Y) :

        reward = np.zeros(9)
        #Y = Y - Y.mean()
        reward[0] = Y[:10, :10].sum()
        reward[1] = Y[10:20, :10].sum()
        reward[2] = Y[20:30, :10].sum()
        reward[3] = Y[0:10, 10:20].sum()
        reward[4] = Y[10:20, 10:20].sum()
        reward[5] = Y[20:30, 10:20].sum()
        reward[6] = Y[0:10, 20:30].sum()
        reward[7] = Y[10:20, 20:30].sum()
        reward[8] = Y[20:30, 20:30].sum()
        reward = reward - np.mean(reward)

        if self.reward_clip :
            reward_argsort = reward.argsort()
            if reward_argsort[-1] == action :
                return 1
            elif reward_argsort[-2] == action or reward_argsort[-3] == action :
                return 0.5
            elif reward_argsort[0] == action :
                return -1
            elif reward_argsort[1] == action or reward_argsort[2] == action :
                return -0.5
            else :
                return 0
        else :
            return float(reward[action])

    def step(self, action, step, episode, render=False, test=False, MAX_STEP=128) :
        terminal = False
        pos_old = self.pos.copy()
        reward, terminal = self.acting(action)
        #print(self.pos)

        # musical state
        mic = get_mic_pos(self.pos)

        h, _, _ = RG.np_generateRir(self.Rdim, self.Sdim, mic,
                                    soundVelocity=self.c, fs=self.fs, reverbTime=self.rt,
                                    nSamples=self.nSamples)
        music, fs = librosa.load("./SA1.WAV")

        frame = []
        for i in range(self.nMic):
            current_frame = np.convolve(h[i, :], music[:self.nSamples], mode='same')
            current_frame = AWGN(current_frame, self.noise_dB)
            frame.append(current_frame)
        frame = np.array(frame)

        Y = srp_phat(frame, W=self.W)
        #Y_norm = np.where(Y==Y.max(), 100, 0)
        Y_norm = Y / np.max(Y)
        Y_max = Y.argmax()
        max_pos = self.Q_array[:,Y_max]
        #cart = self.Q_array.copy()
        #cart[0, :] = cart[0, :] + self.pos[0]
        #cart[1, :] = cart[1, :] + self.pos[1]
        #for i in range(cart.shape[1]):
        #    self.roomgrid[int(cart[0, i] * 10), int(cart[1, i] * 10)] += Y[i]
        Y_2d = np.reshape(Y_norm, [30, 30])
        Y_2d = np.transpose(Y_2d)
        musical_state = Y_2d[np.newaxis, :,:]

        reward += self.reward_action(action, Y_2d)

        # spatial state
        pos_spatial = [int(self.pos[0] * 10), int(self.pos[1] * 10)]
        spatial_state = self.obstacle_room[pos_spatial[0]-15 : pos_spatial[0]+15, pos_spatial[1]-15 : pos_spatial[1]+15]
        spatial_state = np.transpose(spatial_state)
        spatial_state = spatial_state[np.newaxis, :, :]

        state = np.concatenate([musical_state,spatial_state], axis=0)

        if render :
            if step == 1 :
                fig = plt.figure(1)
                ax = plt.subplot(2,2,1)
                ax.scatter(pos_old[0], pos_old[1], marker='$S$', c='b', s=32)
                ax.scatter(self.Sdim[0], self.Sdim[1], marker='$G$', c='b', s=32)
                for i_obstacle in range(self.n_obstacles) :
                    x = self.obstacles[i_obstacle][0] / 10
                    x_length = (self.obstacles[i_obstacle][1] - self.obstacles[i_obstacle][0]) / 10
                    y = self.obstacles[i_obstacle][2] / 10
                    y_length = (self.obstacles[i_obstacle][3] - self.obstacles[i_obstacle][2]) / 10

                    ax.add_patch(patches.Rectangle((x, y), x_length, y_length, edgecolor='blue', facecolor='blue', fill=True))

            ax = plt.subplot(2,2,1)
            ax.scatter(self.pos[0], self.pos[1], marker='o', c='r', s=4)

            #ax.arrow(pos_old[0], pos_old[1], self.pos[0]-pos_old[0],
            #          self.pos[1]-pos_old[1], width = 0.01)

            ax.set_xlim((0, 20))
            ax.set_ylim((20, 0))
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(abs(x1-x0)/abs(y1-y0))
            ax.grid(b=True, which='major', color='k', linestyle='--')

            ax = plt.subplot(2,2,2)
            ax.matshow(Y_2d)

            ax = plt.subplot(2,2,3)
            ax.matshow(spatial_state.squeeze(0))


            plt.draw()
            plt.pause(0.001)
            #ax.remove()

        distance = np.linalg.norm(self.pos-self.Sdim, 2)
        distance_reward = 20 *(self.distance_old - distance)
        if self.reward_clip :
            if distance_reward > 0.3 :
                distance_reward = 1
            else :
                distance_reward = 0
        self.distance_old = distance.copy()
        if distance <= 0.5 :
            terminal = True
            reward += 100
        if terminal :
            if test :
                plt.savefig("./result_test/episode_"+str(episode)+".png")
            else :
                plt.savefig("./result/episode_"+str(episode)+".png")
            plt.clf()
        else :
            if step == MAX_STEP :
                if test:
                    plt.savefig("./result_test/episode_" + str(episode) + ".png")
                else:
                    plt.savefig("./result/episode_" + str(episode) + ".png")
                plt.clf()

        return state, distance_reward + reward, terminal, {'pos' : self.pos, 'distance' : distance}

    def reset(self, render=False) :
        self.roomgrid = np.zeros((self.Rdim[0] * 10, self.Rdim[1] * 10))  # room grid
        self.obstacle_room = np.zeros((self.Rdim[0] * 10, self.Rdim[1] * 10))
        self.distance_old = np.linalg.norm(self.pos-self.Sdim, 2)

        self.obstacles = self.generate_obstacles(self.n_obstacles)
        while True :
            self.Sdim = np.array([np.random.randint(2, 18), np.random.randint(2, 18), 1], dtype=np.float32)
            if not self.obstacle_room[int(self.Sdim[0] * 10), int(self.Sdim[1] * 10)] == 1 :
                break
        while True :
            self.pos = np.array([np.random.randint(2, 18), np.random.randint(2, 18), 1], dtype=np.float32)
            if not self.obstacle_room[int(self.pos[0] * 10), int(self.pos[1] * 10)] == 1 :
                break

        # get musical state
        mic = get_mic_pos(self.pos)
        h, _, _ = RG.np_generateRir(self.Rdim, self.Sdim, mic,
                                    soundVelocity=self.c, fs=self.fs, reverbTime=self.rt,
                                    nSamples=self.nSamples)
        music, fs = librosa.load("./SA1.WAV")

        frame = []
        for i in range(self.nMic):
            current_frame = np.convolve(h[i, :], music[:self.nSamples], mode='same')
            current_frame = AWGN(current_frame, self.noise_dB)
            frame.append(current_frame)
        frame = np.array(frame)

        Y = srp_phat(frame, W=self.W)
        Y = Y / np.max(Y)
        Y_2d = np.reshape(Y, [30, 30])
        Y_2d = np.transpose(Y_2d)
        musical_state = Y_2d[np.newaxis, :, :]

        # get spatial state
        pos_spatial = [int(self.pos[0] * 10), int(self.pos[1] * 10)]
        spatial_state = self.obstacle_room[pos_spatial[0]-15 : pos_spatial[0]+15, pos_spatial[1]-15 : pos_spatial[1]+15]
        spatial_state = spatial_state[np.newaxis, :, :]

        state = np.concatenate([musical_state,spatial_state], axis=0)

        return state

    def close(self) :
        print("clear env")

if __name__ == "__main__" :
    ssl_env = SSL_env()
    for i in range(0, 20):
        ssl_env.step(7, i, 0, render=True)

