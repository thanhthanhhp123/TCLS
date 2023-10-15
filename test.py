import torch
import gym_sokoban
import gym 
from common.ActorCritic import ActorCritic
from common.utils import hwc2chw

#Load model
device = torch.device('cpu')
pretrained_dict = torch.load('model.pt', map_location=device)
actor_critic = ActorCritic((3, 80, 80), 5)
actor_critic.load_state_dict(pretrained_dict)

#Make env
env = gym.make('Boxoban-Val-Basic')
state = env.reset()

done = False

while not done:
    env.render(mode = 'human')
    state = hwc2chw(state, test=True)
    action = actor_critic.select_action(state.unsqueeze(0), test=1)
    next_state, reward, done, _ = env.step(action.item())
    next_state = hwc2chw(next_state, test=True)
    state = next_state
    env.render()
