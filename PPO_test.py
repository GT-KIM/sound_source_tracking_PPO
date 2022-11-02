import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import torch
import torchvision
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch import nn, optim
from datetime import datetime
from time import time
from utils import RunningStats, discount, moving_avg
from SSL_env_test import *
from model import *
OUTPUT_RESULTS_DIR = './'

EP_MAX = 1000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.01 # 0.01 for discrete 0.0 for continuous
LR = 0.0001
BATCH = 16 # 128 for discrete, 8192 for continous
MINIBATCH = 16
EPOCHS = 20
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
SIGMA_FLOOR = 0.0
MAX_STEP=128

MODEL_RESTORE_PATH = 'PPO/SSL/20220228-152421/model-1000.pth'
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

            entropy = dist.entropy()
            pol_entpen = -ENTROPY_BETA * entropy.mean()
            losses['entropy'].append(pol_entpen.item())

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            losses['loss'].append(loss.item())

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
            e = self.anet(state).probs
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
    env_for_srp = SSL_env_for_srp_phat()
    ppo = PPO(env, LOG_DIR, gpu=True)
    if MODEL_RESTORE_PATH is not None :
        ppo.restore_model(MODEL_RESTORE_PATH)
        print("Restore Done")
    # Run trained policy
    if not os.path.isdir('./result_test/') :
        os.makedirs('./result_test/')
    text = open('./result_test/result.txt', 'w')

    num_finish = 0
    movings = list()
    rewards = list()
    for episode in range(100) :
        s, distance_raw, a = env_for_srp.reset(episode)
        step = 0
        ep_r, ep_t = 0, 0
        moving_distance = 0
        while True :
            step += 1
            s, r, terminal, _, moving_distance, a = env_for_srp.step(a, step, episode, moving_distance, render=True, test=True, MAX_STEP=MAX_STEP, name='srp')
            ep_r += r

            #print("Episode : %i step : %d action : %d reward : %f distance : %f" % (episode, step, a, r, moving_distance))

            if terminal or step == MAX_STEP :
                ep_r = ep_r / step
                #print('Episode : %i' % episode, "| Reward : %.2f" % ep_r, '| Steps : %i' % ep_t)
                if terminal :
                    num_finish += 1
                break
        movings.append(moving_distance - distance_raw)
        rewards.append(ep_r)
    print("SRP-PHAT accuracy : {0} mean moving distance : {1} mean reward : {2}".format(num_finish/100, np.mean(movings), np.mean(rewards)))
    text.write("SRP-PHAT accuracy : {0} mean moving distance : {1} mean reward : {2}\n".format(num_finish/100, np.mean(movings), np.mean(rewards)))


    num_finish = 0
    movings = list()
    rewards = list()
    for episode in range(100) :
        s, distance_raw = env.reset(episode)
        step = 0
        ep_r, ep_t = 0, 0
        moving_distance = 0
        while True :
            step += 1
            a = np.random.randint(8)
            s, r, terminal, _, moving_distance = env.step(a, step, episode, moving_distance, render=True, test=True, MAX_STEP=MAX_STEP, name='rand')
            ep_r += r

            #print("Episode : %i step : %d action : %d reward : %f distance : %f" % (episode, step, a, r, moving_distance))

            if terminal or step == MAX_STEP :
                ep_r = ep_r / step
                #print('Episode : %i' % episode, "| Reward : %.2f" % ep_r, '| Steps : %i' % ep_t)
                if terminal :
                    num_finish += 1
                break
        movings.append(moving_distance - distance_raw)
        rewards.append(ep_r)
    print("Random walk accuracy : {0} mean moving distance : {1} mean reward : {2}".format(num_finish/100, np.mean(movings), np.mean(rewards)))
    text.write("Random walk accuracy : {0} mean moving distance : {1} mean reward : {2}\n".format(num_finish/100, np.mean(movings), np.mean(rewards)))

    num_finish = 0
    movings = list()
    rewards = list()
    for episode in range(100) :
        s, distance_raw = env.reset(episode)
        step = 0
        ep_r, ep_t = 0, 0

        moving_distance = 0
        while True :
            step += 1
            a, v = ppo.evaluate_state(s, stochastic=False)
            s, r, terminal, _, moving_distance = env.step(a, step, episode, moving_distance, render=True, test=True, MAX_STEP=MAX_STEP, name='proposed')
            ep_r += r
            #print("Episode : %i step : %d action : %d reward : %f distance : %f" % (episode, step, a, r, moving_distance))

            if terminal or step == MAX_STEP :
                ep_r = ep_r / step
                #print('Episode : %i' % episode, "| Reward : %.2f" % ep_r, '| Steps : %i' % ep_t)
                if terminal :
                    num_finish += 1
                break
        movings.append(moving_distance - distance_raw)
        rewards.append(ep_r)
    print("PPO accuracy : {0} mean moving distance : {1} mean reward : {2}".format(num_finish/100, np.mean(movings) ,np.mean(rewards)))
    text.write("PPO accuracy : {0} mean moving distance : {1} mean reward : {2}\n".format(num_finish/100, np.mean(movings) ,np.mean(rewards)))
    text.close()


