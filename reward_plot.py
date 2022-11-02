import numpy as np
import matplotlib.pyplot as plt
d = np.load("result_20210818/avg_reward.npz")
avg_reward_list = d['avg_reward_list']
reward_x = d['reward_x']
avg_reward_trend = d['avg_reward_trend']

def moving_avg(x , n=10) :
    return np.convolve(x, np.ones(n), 'same') / n

trend = moving_avg(avg_reward_list, 20)
plt.plot(reward_x[:], avg_reward_list[:])
plt.plot(reward_x[:], trend[:])
plt.xlabel("episode")
plt.ylabel("avg. reward")
plt.show()
