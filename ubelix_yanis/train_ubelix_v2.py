import numpy as np
import torch as th
import torch.nn.functional as F
from copy import deepcopy
import simglucose
import gym
from gym.wrappers import FlattenObservation
from tqdm import tqdm
import os

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device=th.device('cuda' if th.cuda.is_available() else 'cpu'), max_size=500):
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
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = th.nn.Linear(state_dim, 200)
        self.l2 = th.nn.Linear(200, 200)
        self.l3 = th.nn.Linear(200, 10)
        self.l4 = th.nn.Linear(10, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = th.sigmoid(self.l4(a))
        return 1.8 * a + 0.2 # scale to [0.2, 2.0]

class Critic(th.nn.Module): # state + action -> Q(s,a) (Q-Network)
    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()
        self.l1 = th.nn.Linear(state_dimension + action_dimension, 200)
        self.l2 = th.nn.Linear(200, 200)
        self.l3 = th.nn.Linear(200, 10)
        self.l4 = th.nn.Linear(10, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(th.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        q = self.l4(q)
        return q

# Define DDPG Agent
class DDPGAgent(object):
    def __init__(self, env, device=th.device('cuda' if th.cuda.is_available() else 'cpu'), discount=0.9, tau=0.01):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = device
        self.discount = discount
        self.tau = tau
        self.is_pretrained = False
        # Actor and Actor target
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_target = deepcopy(self.actor)
        # Critic and Critic target
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.critic_target = deepcopy(self.critic)
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.device)

    def save_agent(self, path=None):
        if path is None:
            path = 'agent_state'
        models = [self.actor, self.actor_target, self.actor_optimizer, self.critic, self.critic_target, self.critic_optimizer]
        fnames = ['actor', 'actor_target', 'actor_optimizer', 'critic', 'critic_target', 'critic_optimizer']
        if not os.path.exists(path):
            os.makedirs(path)
        for m, f in zip(models, fnames):
            th.save(m.state_dict(), os.path.join(path, f+'.pt'))
        print(f'Agent saved to folder {path}')

    def load_agent(self, path=None):
        if path is None:
            path = 'agent_state'
        models = [self.actor, self.actor_target, self.actor_optimizer, self.critic, self.critic_target, self.critic_optimizer]
        fnames = ['actor', 'actor_target', 'actor_optimizer', 'critic', 'critic_target', 'critic_optimizer']
        for m, f in zip(models, fnames):
            m = th.load(os.path.join(path, f+'.pt'), map_location='cpu')
        print(f'Agent loaded from folder {path}')
    
    def soft_update(self):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state): # Actor selects action based on current state
        return self.actor(th.tensor(state, dtype=th.float32).to(self.device)).detach().cpu().numpy()

    def _train(self, batch_size, target_update_period, max_iter, max_patience, path):
        CHO_idx = int(2*self.state_dim/3 - 1)
        critic_training_loss = []
        min_critic_loss = float('inf')
        patience = max_patience
        for it in tqdm(range(max_iter)):
            # Sample one episode and add it to the replay buffer
            state, info = self.env.reset()
            reset_time = info['time']
            done = False
            last_meal = state[CHO_idx] # XXX
            while not done and (info['time'] - reset_time).days < 4: # 4 days
                if last_meal > state[CHO_idx]: # First step after meal
                    start_state = state
                    start_time = info['time']
                    noise = np.random.normal(0, 0.3, (3,))
                    bolus_action = self.select_action(start_state) + noise
                    last_meal = state[CHO_idx] #### XXX
                    state, reward, done, _, info = self.env.step(bolus_action)
                    reward_sum = reward
                    while not done and state[CHO_idx] == 0 and (info['time'] - start_time).total_seconds() < 5*3600:
                        action = [0, 0, 0]
                        last_meal = state[CHO_idx] #### XXX
                        state, reward, done, _, info = self.env.step(action)
                        reward_sum += reward
                    reward_sum /= (info['time'] - start_time).total_seconds() / 60
                    next_state = state
                    self.replay_buffer.store(start_state, bolus_action, reward_sum, next_state, done)
                    # print(f'Episode (reward: {reward_sum}) stored to memory ({self.replay_buffer.size})')
 
                else:
                    action = [0, 0, 0]
                    last_meal = state[CHO_idx] #### XXX
                    state, _, done, _, info = self.env.step(action)

                
            
            if self.replay_buffer.size >= batch_size:
                # Sample replay buffer
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
                # Compute the target Q value
                target_q = self.critic_target(next_states, self.actor_target(next_states))
                target_q = rewards + (self.discount * target_q).detach()
                # target_q = rewards + ((not dones) * self.discount * target_q).detach() # XXX: ?????????????
                # Get current Q estimate
                current_q = self.critic(states, actions)
                # Compute critic loss
                critic_loss = F.mse_loss(current_q, target_q)
                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states)).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # Save training loss   
                if critic_loss < min_critic_loss:
                    self.save_agent(path=path)
                    min_critic_loss = critic_loss
                    patience = max_patience
                critic_training_loss.append([critic_loss.item(), actor_loss.item()])
                
                # Update target networks
                if it % target_update_period == 0:
                    self.soft_update()

                if it % 5 == 0:
                    print(f'Iteration: {it+1}, Critic loss: {critic_loss.item():.3f} (min: {min_critic_loss:.3f}), Patience left: {patience}')

                # Convergence check
                patience -= 1
                if patience == 0:
                    print('Critic converged...')
                    break

        return critic_training_loss

    def general_training(self, batch_size=32, target_update_period=100, max_iter=5000, max_patience=100, path='agent_state', filename='general_training_loss.json'):
        g_critic_loss = self._train(batch_size=batch_size, target_update_period=target_update_period, max_iter=max_iter, max_patience=max_patience, path=path)
        self.is_pretrained = True

        # if filename is not None:
        #     file = os.path.join(path, filename)
        #     with open(file, 'w') as f:
        #         json.dump(g_critic_loss, f)
        #     print(f'Critic loss saved to {file}')

        return g_critic_loss

    def personalized_training(self, individual_env, batch_size=32, target_update_period=100, max_iter=1000, max_patience=50, path='agent_state_finetuned', filename='general_training_loss.json'):
        assert self.is_pretrained == True, 'Agent must be pretrained before finetuning'
        assert individual_env.observation_space.shape[0] == self.state_dim, 'State dimension mismatch'
        assert individual_env.action_space.shape[0] == self.action_dim, 'Action dimension mismatch'

        # Change to individual environment (but keep old one)
        temp_env = self.env
        self.env = individual_env

        ft_critic_loss = self._train(batch_size=batch_size, target_update_period=target_update_period, max_iter=max_iter, max_patience=max_patience, path=path)

        # Change back to original environment
        self.env = temp_env

        # if filename is not None:
        #     file = os.path.join(path, filename)
        #     with open(file, 'w') as f:
        #         json.dump(ft_critic_loss, f)
        #     print(f'Critic loss saved to {file}')

        return ft_critic_loss

    def evaluate_policy(self, individual_env=None, max_iter=1000, render=False):
        CGM_idx = int(self.state_dim/3 - 1)
        CHO_idx = int(2*self.state_dim/3 - 1)

        # initialize metrics
        actor_output = []
        in_range = {'target': 0, 'hypo': 0, 'hyper': 0, 'total': 0}
        metrics = dict()

        # Change to individual environment (but keep old one)
        if individual_env is not None:
            assert individual_env.observation_space.shape[0] == self.state_dim, 'State dimension mismatch'
            assert individual_env.action_space.shape[0] == self.action_dim, 'Action dimension mismatch'
            temp_env = self.env
            self.env = individual_env

        state, info = self.env.reset()
        last_meal = 0
        for t in tqdm(range(max_iter)):
            
            if render:
                self.env.render(mode='human')

            if last_meal > state[CHO_idx]: # Bolus given
                action = self.select_action(state)
                actor_output.append((info['time'], action))
            else:
                action = [0, 0, 0]
            last_meal = state[CHO_idx]
            state, _, done, _, info = self.env.step(action)

            if state[CGM_idx] < 70:
                in_range['hypo'] += 1
            elif state[CGM_idx] > 180:
                in_range['hyper'] += 1
            else:
                in_range['target'] += 1
            in_range['total'] += 1

            if done:         
                print(f'Episode finished after {t+1} timesteps (patient died).')
                break

        print('Episode finished.')

        metrics['actor_output'] = actor_output
        metrics['TIR'] = in_range['target']/in_range['total']
        metrics['hypo'] = in_range['hypo']/in_range['total']
        metrics['hyper'] = in_range['hyper']/in_range['total']

        if individual_env is not None:
            # Change back to original environment
            self.env = temp_env

        return metrics

# Define reward function based on paper
def custom_reward(BG_history):
    BG = BG_history[-1]
    # BG: blood glucose level
    # Hypoglycemia: BG < 70 mg/dL
    if 30 <= BG and BG < 70:
        return -1.5
    # Normoglycemia: 70 mg/dL < BG < 180 mg/dL
    elif 70 <= BG and BG <= 180:
        return 0.5
    # Hyperglycemia: BG > 180 mg/dL
    elif 180 < BG and BG <= 300:
        return -0.8
    # elif 300 < BG and BG <= 350:
    #     return -1
    # Other cases
    else:
        return -10

def make_env(id: str, patient_name: str, history_length=6, reward_function=custom_reward, print_space=True, flatten=True):
    gym.envs.register(
        id=id,
        entry_point='simglucose.envs:T1DSimEnvBolus',
        kwargs={'patient_name': [patient_name],
            'history_length': history_length, 'reward_fun': reward_function,
            'enable_meal': True})

    env = gym.make(id)

    if print_space:
        print('State space:\n', env.observation_space)
        print('Action space:\n', env.action_space)

    if flatten:
        env = FlattenObservation(env)

    return env

def main():

    average_patient = make_env('simglucose_average', 'average_adolescent', print_space=False)
    agent = DDPGAgent(average_patient)

    critic_loss = agent.general_training(path='agent_state_test')

    import matplotlib.pyplot as plt

    plt.plot([x[0] for x in critic_loss])
    plt.savefig('critic_loss.png')

if __name__ == '__main__':
    main()


