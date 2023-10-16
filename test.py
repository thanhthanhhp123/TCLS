import torch
import gym_sokoban
import gym 
from common.ActorCritic import ActorCritic
from common.utils import hwc2chw
from common.test_the_agent import test_the_agent
import time
from IPython import display
from matplotlib import pyplot as plt

#Load model
device = torch.device('cpu')
pretrained_dict = torch.load('model.pt', map_location=device)
actor_critic = ActorCritic((3, 80, 80), 5)
actor_critic.load_state_dict(pretrained_dict)

#Make env
env = gym.make('Boxoban-Val-Basic')
for i in range(100):
    state = env.reset()
    state = hwc2chw(state, test=True)
    action = actor_critic.select_action(state.unsqueeze(0), test=1)
    next_state, reward, done, _ = env.step(action.item())
    next_state = hwc2chw(next_state, test=True)

    i = 1
    while not done:
        state = next_state
        with torch.no_grad():
            action = actor_critic.select_action(state.unsqueeze(0), test=1)
        next_state, reward, done, _ = env.step(action.item())
        next_state = hwc2chw(next_state, test=True)
        plt.imshow(next_state.reshape(80, 80, 3))
        display.clear_output(wait=True)
        plt.show()