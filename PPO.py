import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import torch
import torchvision
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch import nn, optim
from datetime import datetime
from time import time
from utils import RunningStats, discount, moving_avg
from SSL_env import *
from model import *
OUTPUT_RESULTS_DIR = './'

EP_MAX = 1000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.01 # 0.01 for discrete 0.0 for continuous
LR = 0.001
BATCH = 16 # 128 for discrete, 8192 for continous
MINIBATCH = 16
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
SIGMA_FLOOR = 0.0
MAX_STEP=128

MODEL_RESTORE_PATH = None
if not os.path.isdir('./result/') :
    os.makedirs('./result')

class PPO(object) :
    def __init__(self, environment, LOG_DIR, gpu=True, graycale=False) :
        self.discrete = True
        self.s_dim, self.a_dim = np.where(environment.Rdim==5, 1, 30), 9

        # model
        self.anet_old = Action_Network(a_dim=self.a_dim, SIGMA_FLOOR=SIGMA_FLOOR).cuda()
        self.anet = Action_Network(a_dim=self.a_dim, SIGMA_FLOOR=SIGMA_FLOOR).cuda()

        self.cnet_old = Critic_Network().cuda()
        self.cnet = Critic_Network().cuda()

        # loss
        self.MseLoss = nn.MSELoss()
        # optimizer
        self.optimizer = optim.Adam(list(self.anet.parameters()) + list(self.cnet.parameters()),lr=LR, weight_decay=1e-3)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,[100, 200, 300, 400, 500], gamma=0.1)

    def save_model(self, model_path, step=0) :
        if not os.path.isdir(model_path) :
            os.makedirs(model_path)
        state_dict = {
            'anet': self.anet.state_dict(),
            'cnet': self.cnet.state_dict(),
        }
        torch.save(state_dict, os.path.join(model_path, "model-{}.pth".format(step)))

    def restore_model(self, model_path) :
        ckpt = torch.load(model_path)
        self.anet.load_state_dict(ckpt['anet'])
        self.cnet.load_state_dict(ckpt['cnet'])
        self.anet_old.load_state_dict(ckpt['anet'])
        self.cnet_old.load_state_dict(ckpt['cnet'])

    def update(self, s, a, r, adv, global_step):
        epsilon_decay = 0.1#float(np.clip(0.1-1e5*global_step, 0.01, 0.1))

        self.anet_old.load_state_dict(self.anet.state_dict())
        self.cnet_old.load_state_dict(self.cnet.state_dict())

        losses = {'loss_pi' : list(), 'loss_vf' : list(), 'entropy' : list(), 'loss' : list()}

        for _ in range(EPOCHS) :
            dist = self.anet(torch.tensor(s).float().cuda())
            dist_old = self.anet_old(torch.tensor(s).float().cuda())
            vf = self.cnet(torch.tensor(s).float().cuda())
            vf_old = self.cnet_old(torch.tensor(s).float().cuda())


            ratio = torch.exp(dist.log_prob(torch.tensor(a).squeeze(1).float().cuda())
                              - dist_old.log_prob(torch.tensor(a).squeeze(1).float().cuda()))
            ratio = torch.clamp(ratio, 0, 10).unsqueeze(1)
            surr1 = torch.tensor(adv).float().cuda() * ratio
            surr2 = torch.tensor(adv).float().cuda() * torch.clamp(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
            loss_pi = - torch.min(surr1, surr2).mean()
            losses['loss_pi'].append(loss_pi.item())

            clipped_value_estimate = vf_old + torch.clamp(vf - vf_old, -epsilon_decay, epsilon_decay)
            loss_vf1 = self.MseLoss(clipped_value_estimate.unsqueeze(1), torch.tensor(r).float().cuda())
            loss_vf2 = self.MseLoss(vf.unsqueeze(1), torch.tensor(r).float().cuda())
            loss_vf = 0.5 * torch.maximum(loss_vf1, loss_vf2)
            losses['loss_vf'].append(loss_vf.item())

            entropy = dist.entropy().mean()
            pol_entpen = -ENTROPY_BETA * entropy
            losses['entropy'].append(pol_entpen.item())

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            losses['loss'].append(loss.item())
            print("loss : %f loss_pi : %f loss_vf : %f entropy : %f" % (loss.item(),loss_pi.item(),loss_vf.item(),pol_entpen.item()))


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            global_step += 1
        return losses

    def evaluate_state(self, state, stochastic=True) :
        state = torch.tensor(state).float().unsqueeze(0).cuda()
        if stochastic :
            action = self.anet(state).sample()
            value = self.cnet(state)
        else :
            action = torch.argmax(self.anet(state).probs, dim=1)
            value = self.cnet(state)
        action = action.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        return action, value

if __name__ == '__main__' :
    ENVIROMENT = 'SSL'
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO", ENVIROMENT, TIMESTAMP)

    env = SSL_env()
    ppo = PPO(env, LOG_DIR, gpu=True)

    if MODEL_RESTORE_PATH is not None :
        ppo.restore_model(MODEL_RESTORE_PATH)
        print("Restore Done")
    t, terminal = 0 , False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = list(), list(), list(), list(), list()
    rolling_r = RunningStats()

    avg_reward_list = list()
    global_step = 0
    for episode in range(EP_MAX + 1) :
        s = env.reset()
        ep_r, ep_t, ep_a = 0, 0, list()
        step = 0
        next_action = 4
        while True :
            step += 1
            a, v = ppo.evaluate_state(s)
            # update ppo
            if t == BATCH or (terminal and t < BATCH) :
                # Normalize rewards
                rewards = np.array(buffer_r)
                rolling_r.update(rewards)
                rewards = np.clip(rewards / rolling_r.std, -10, 10)
                #rewards = (rewards - rolling_r.mean / rolling_r.std)

                v_final = [v * (1 - terminal)]
                values = np.array(buffer_v + v_final)
                terminals = np.array(buffer_terminal + [terminal])

                # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                delta = rewards[:,np.newaxis] + GAMMA * values[1:] * (1 - terminals[1:])[:, np.newaxis] - values[:-1]
                advantage = discount(delta, GAMMA * LAMBDA, terminals)
                returns = advantage + np.array(buffer_v)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                bs, ba, br, badv = np.reshape(buffer_s, (t, 2, 30, 30)), np.vstack(buffer_a), np.vstack(returns), np.vstack(advantage)

                losses = ppo.update(bs, ba, br, badv, global_step)
                buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
                t = 0
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(v)
            buffer_terminal.append(terminal)
            ep_a.append(a)

            s, r, terminal, properties = env.step(a, step, episode, render=True, MAX_STEP=MAX_STEP)
            if step == 1 :
                distance = properties['distance']
                pos = properties['pos']
                reward_dist = 0
            else :
                distance_old = distance
                pos_old = pos
                distance = properties['distance']
                pos = properties['pos']
                reward_dist = distance_old - distance

            print("Episode : %i step : %d action : %d reward : %f distance : %f" % (episode, step, a, r, distance))

            buffer_r.append(r)

            ep_r += r
            ep_t += 1
            t += 1

            if terminal or step == MAX_STEP :
                avg_reward = ep_r / step
                avg_reward_list.append(avg_reward)
                ppo.lr_scheduler.step()
                print('Episode : %i' % episode, "| Reward : %.2f" % avg_reward, '| Steps : %i' % ep_t)
                fig = plt.figure(2)
                plt.clf()
                plt.title("SSL1")
                plt.plot(np.arange(len(avg_reward_list)), avg_reward_list)
                plt.draw()
                plt.pause(0.001)

                # Save the model
                if episode % 100 == 0 and episode > 0 :
                    ppo.save_model(LOG_DIR, episode)
                    print('Saved model at episode', episode, ' in ', LOG_DIR)
                break
    avg_reward_list = np.array(avg_reward_list)
    reward_x = np.arange(EP_MAX+1)
    avg_reward_trend = moving_avg(avg_reward_list, 20)
    np.savez("./result/avg_reward.npz", avg_reward_list = avg_reward_list,
             reward_x = reward_x, avg_reward_trend = avg_reward_trend)
    plt.plot(reward_x, avg_reward_list)
    plt.plot(reward_x, avg_reward_trend)
    plt.show()
    env.close()

    # Run trained policy
    env = SSL_env()
    if not os.path.isdir('./result_test/') :
        os.makedirs('./result_test/')
    while True :
        s = env.reset()
        ep_r, ep_t = 0, 0
        step = 0
        episode = 0
        while True :
            step += 1
            a, v = ppo.evaluate_state(s, stochastic=False)
            s, r, terminal, _ = env.step(a, step, episode, render=True, test=True)
            ep_r += r
            ep_t += 1
            if terminal or ep_t == MAX_STEP :
                print('Reward : %.2f' % ep_r, '| Steps : %i' % ep_t)
                break
        episode += 1