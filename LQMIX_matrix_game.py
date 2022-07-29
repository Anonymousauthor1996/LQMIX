# a simple version of LQMIX in one step matrix game
# for simplicity, we omit the projection and auxiliary agent netowrk (since there only exists one state)
from ntpath import join
import torch as th
import torch.nn as nn
from torch.optim import RMSprop
import random
import torch.nn.functional as F
from math import *

explore=True
factorization = "owqmix"           # qmix, vdn, owqmix, lqmix

# env
matrix = th.tensor([[8,-12,-12],[-12,6,6],[-12,6,6]]).float()
mat_size = 3
episodes = 10000
batch_size = 32
lr = 0.001
epsl = 1
weight = 0.1
sigma_threshold = 0.5
random_reward = False
random_prob = 0.5
random_reward_value = [12, 0]

Q_matrix = th.randint(-10,10,(mat_size, mat_size)).float()
Q_matrix_cal = th.randint(-10,10,(mat_size, mat_size)).float()
### lqmix specific
visition, h = th.zeros(1, 2, 3).long(), th.zeros(1, 2, 3)

# agent
class Q_mat(nn.Module):
    def __init__(self, mat_size):
        super(Q_mat, self).__init__()
        self.Q_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, mat_size))
    def forward(self, inp):
        return self.Q_net(inp)
# qmixer
class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        self.mixer_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
        self.v_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, inp, q_1, q_2):
        return th.abs(self.mixer_net(inp))[0] * q_1 + th.abs(self.mixer_net(inp))[1] * q_2 + self.v_net(inp)
# lqmix specific
class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        self.r_net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        self.s_net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, inputs):
        r = self.r_net(inputs)
        sigma = self.s_net(inputs)
        return r, sigma
    
def look_up_temp(visition,action):
    action = action.unsqueeze(-1)
    visit = th.gather(visition[0], dim=-1, index=action).squeeze(-1)
    temp = 0.999 ** visit.min(-1)[0].item()
    visition[0].scatter_add_(-1, action.long(), th.ones_like(action))
    return th.tensor(temp).expand_as(action)

def look_up_h(h, action, reward, is_sr, predict_reward, is_greedy):
    h_inm = th.gather(h[0], dim=-1, index=action.unsqueeze(-1))
    rewards = reward.unsqueeze(-1).unsqueeze(-1).repeat(2,1)
    if is_sr and is_greedy:
        h1 = predict_reward.expand_as(rewards)
    elif is_sr:
        h1 = h_inm
    else:
        h1 = th.where(rewards > h_inm, rewards, h_inm)
    h[0].scatter_(-1, action.unsqueeze(-1).long(), h1)
    return th.tensor(h1).expand_as(action.unsqueeze(-1))

def transfrom_onehot(joint_ac):
    joint_ac = joint_ac.unsqueeze(-1)
    y_onehot = joint_ac.new(*joint_ac.shape[:-1], 3).zero_()
    y_onehot.scatter_(-1, joint_ac.long(), 1)
    return y_onehot.view(-1).float()

q_in = th.ones(4)
agent_1 = Q_mat(mat_size)
agent_2 = Q_mat(mat_size)

params = list(agent_1.parameters())
params += list(agent_2.parameters())
if factorization in ["qmix","owqmix","lqmix"]:
    mixer = Mixer()
    params += list(mixer.parameters())
if factorization == "lqmix":
    reward_predictor = RewardPredictor()
    params += list(reward_predictor.parameters())

optimiser = RMSprop(params=params, lr=lr)

for ep in range(episodes):
    loss, reward_loss = 0, 0
    for i in range(batch_size):
        q_out1 = agent_1.forward(q_in)
        q_out2 = agent_2.forward(q_in)
        rand = random.random()
        if rand < epsl:                 # exploration
            ac_1 = th.randint(0,mat_size,()).long()
            ac_2 = th.randint(0,mat_size,()).long()
        else:                           
            ac_1 = q_out1.max(0)[1].long()
            ac_2 = q_out2.max(0)[1].long()
        if random_reward:
            if ac_1 == 1 and ac_2 == 1:
                if random.random() < random_prob:
                    Gt = th.tensor(random_reward_value[0]).float()
                else:
                    Gt = th.tensor(random_reward_value[1]).float()
            else:
                Gt = matrix[ac_1, ac_2]
        else:
            Gt = matrix[ac_1, ac_2]
        if factorization in ["qmix","owqmix","lqmix"]:
            q_tot = mixer(q_in, q_out1[ac_1], q_out2[ac_2])
        elif factorization == "vdn":
            q_tot = q_out1[ac_1] + q_out2[ac_2]

        error = Gt - q_tot
        if factorization == "owqmix":
            w = th.where(error > 0, th.ones_like(error), th.ones_like(error)*weight)
            loss +=  (w * error) ** 2     

        elif factorization == "lqmix":
            joint_ac, joint_greedy_ac = th.cat((ac_1.unsqueeze(0), ac_2.unsqueeze(0))), th.cat((q_out1.max(0)[1].long().unsqueeze(0), q_out2.max(0)[1].long().unsqueeze(0)))
            joint_ac_onehot = transfrom_onehot(joint_ac)
            inputs = th.cat((q_in, joint_ac_onehot))
            r, sigma = reward_predictor(inputs)
            is_greedy = th.where(joint_ac == joint_greedy_ac, True, False).min()
            temp = look_up_temp(visition, joint_ac)
            random_z = th.rand_like(temp)
            leniency = 1 - e**(-2*temp)
            not_forgive = random_z > leniency
            tot_forgive = not_forgive.min(0)[0].unsqueeze(-1).detach()
            loss += ((r - Gt)**2/(2*sigma**2) + 0.5*th.log(sigma**2)).sum()
            is_sr = th.where((abs(sigma) > sigma_threshold) & tot_forgive, th.ones_like(r), th.zeros_like(r))
            min_h = look_up_h(h, joint_ac, Gt, is_sr, r, is_greedy).min()

            new_Gt = th.where((is_sr==1) & ~is_greedy, min_h, Gt)   
            new_Gt = th.where(((Gt < min_h) & ~is_greedy), min_h, new_Gt) 
            error = new_Gt - q_tot
            loss += ((error) ** 2).sum()
            
        else:
            loss += (error) ** 2       

    optimiser.zero_grad()
    loss = loss/batch_size
    loss.backward()
    optimiser.step()

    if ep % 500 == 0:
        print("episode:",ep)

        for x in range(mat_size):
            for y in range(mat_size):
                if factorization in ["qmix","owqmix", "lqmix"]:
                    Q_matrix[x][y] = mixer(q_in, q_out1[x], q_out2[y])
                elif factorization == "vdn":
                    Q_matrix[x][y] = q_out1[x] + q_out2[y]
        
        print("Greedy Action:", q_out1.max(0)[1].item()+1, q_out2.max(0)[1].item()+1)
        print("Joint Action Value\n", Q_matrix)
