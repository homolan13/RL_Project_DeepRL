print('HEEELLO WOOOOORLD')

import numpy as np
import torch as th
import torch.nn.functional as F
from copy import deepcopy
import gym

print('HELLO WORLD')

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.size = 0
        self.ptr = 0
        self.state_buffer = np.zeros((max_size, state_dim))
        self.action_buffer = np.zeros((max_size, action_dim))
        self.next_state_buffer = np.zeros((max_size, state_dim))
        self.reward_buffer = np.zeros((max_size, 1))
        self.done_buffer = np.zeros((max_size, 1))

    def store(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.next_state_buffer[self.ptr] = next_state
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            th.tensor(self.state_buffer[idx], dtype=th.float32).to(self.device),
            th.tensor(self.action_buffer[idx], dtype=th.float32).to(self.device),
            th.tensor(self.reward_buffer[idx], dtype=th.float32).to(self.device),
            th.tensor(self.next_state_buffer[idx], dtype=th.float32).to(self.device),
            th.tensor(self.done_buffer[idx], dtype=th.float32).to(self.device)
        )

# Define Actor and Critic networks
class Actor(th.nn.Module): # state -> action
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = th.nn.Linear(state_dim, 400)
        self.l2 = th.nn.Linear(400, 300)
        self.l3 = th.nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * th.tanh(self.l3(a))

class Critic(th.nn.Module): # state + action -> Q(s,a) (Q-Network)
    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()
        self.l1 = th.nn.Linear(state_dimension + action_dimension, 400)
        self.l2 = th.nn.Linear(400, 300)
        self.l3 = th.nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(th.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)

# Define Agents
class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):
        self.device = device
        self.discount = discount
        self.tau = tau
        # Actor and Actor target
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters())
        self.actor_target = deepcopy(self.actor)
        # Critic and Critic target
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters())
        self.critic_target = deepcopy(self.critic)
        
    def select_action(self, state): # Actor selects action based on current state
        state = th.FloatTensor(state).reshape(1, -1).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    @staticmethod
    def soft_update(local_model, target_model, tau): # Soft update of target parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, replay_buffer, batch_size=100):
        # Sample from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        # Compute the target Q value
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (done * self.discount * target_q).detach()
        # Get current Q estimate
        current_q = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update the target models
        DDPGAgent.soft_update(self.critic, self.critic_target, self.tau)
        DDPGAgent.soft_update(self.actor, self.actor_target, self.tau)


def main():
    print('2')
    gym.envs.register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002'}
    )
    print('3')
    env = gym.make('simglucose-adolescent2-v0')
    print('4')
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPGAgent(state_dimension, action_dimension, max_action, device)
    memory = ReplayBuffer(state_dimension, action_dimension, device)

    score_history = []
    num_episodes = 10000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.store(state, action, next_state, reward, done)
            state = next_state
            score += reward
        score_history.append(score)
        agent.train(memory)
        print(f'Episode: {episode} Score: {score}')

    with open('score_history.txt', 'w') as f:
        for i, score in enumerate(score_history):
            f.write((i+1, score))


if __name__ == '__main__':
    print('Hello World 1')
    main()