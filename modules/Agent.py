import numpy as np
import torch as th
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
import os
import json

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

# Define Actor and Critic networks (dimensions given in paper)
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
        return 0.8 * a + 0.6 # scale to [0.2, 2.0]

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
        self.is_pretrained = False # Is set to True when calling self.generalized_training and has to be True for self.personalized_training
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
            m = th.load(os.path.join(path, f+'.pt'), map_location=self.device)
        print(f'Agent loaded from folder {path}')
    
    def soft_update(self):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state): # Actor selects action based on current state
        return self.actor(th.tensor(state, dtype=th.float32).to(self.device)).detach().cpu().numpy()

    def get_env_info(self, print_info=False):
        env_id = self.env.unwrapped.spec.id
        if print_info:
            print('Environment ID: ', env_id)

        return env_id

    def change_env(self, new_env):
        assert new_env.observation_space.shape[0] == self.state_dim, 'State dimension mismatch'
        assert new_env.action_space.shape[0] == self.action_dim, 'Action dimension mismatch'

        old_env_info = self.get_env_info()
        self.env = new_env
        print(f'Environment changed from {old_env_info} to {self.get_env_info()}')

    def _train(self, batch_size: int, target_update_period: int, iter: list, max_patience: int, add_noise: bool, path=None):
        CHO_idx = int(2*self.state_dim/3 - 1)
        min_iter = iter[0]
        max_iter = iter[1]
        training_loss = []
        min_critic_loss = float('inf')
        patience = max_patience
        for it in tqdm(range(max_iter)):
            # Sample one episode and add it to the replay buffer
            state, info = self.env.reset()
            reset_time = info['time']
            done = False
            last_meal = state[CHO_idx]
            while not done and (info['time'] - reset_time).days < 4: # 4 days
                if last_meal > state[CHO_idx]: # First step after meal -> bolus injection
                    start_state = state
                    start_time = info['time']
                    if add_noise: # Add noise when doing general training
                        noise = np.random.normal(0, 0.1, (self.action_dim,))
                    else: # No noise when doing personalized training
                        noise = np.zeros(shape=(self.action_dim,))
                    bolus_action = self.select_action(start_state) + noise
                    last_meal = state[CHO_idx]
                    state, reward, done, _, info = self.env.step(bolus_action) # inject bolus
                    # Start of reward calculation
                    reward_sum = reward
                    while not done and state[CHO_idx] == 0 and (info['time'] - start_time).total_seconds() < 5*3600:
                        action = [0, 0, 0]
                        last_meal = state[CHO_idx]
                        state, reward, done, _, info = self.env.step(action)
                        reward_sum += reward
                    reward_sum /= (info['time'] - start_time).total_seconds() / 60
                    # End of reward calculation
                    next_state = state
                    self.replay_buffer.store(start_state, bolus_action, reward_sum, next_state, done) # add episode to replay buffer
 
                else: # If no bolus necessary -> action is 0
                    action = [0, 0, 0]
                    last_meal = state[CHO_idx]
                    state, _, done, _, info = self.env.step(action)

                
            
            if self.replay_buffer.size >= batch_size: # Train only if replay buffer has enough samples
                # Sample replay buffer
                states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
                # Compute the target Q value
                target_q = self.critic_target(next_states, self.actor_target(next_states))
                target_q = rewards + (self.discount * target_q).detach()
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
                else:
                    patience -= 1
                training_loss.append([critic_loss.item(), actor_loss.item()])
                
                # Update target networks
                if it % target_update_period == 0:
                    self.soft_update()

                # Convergence check
                if patience == 0 and it > min_iter:
                    print('Critic converged...')
                    break

        return training_loss

    def general_training(self, batch_size=32, target_update_period=100, iter=[800, 3000], max_patience=350, path='agent_state'):
        g_training_loss = self._train(batch_size=batch_size, target_update_period=target_update_period, iter=iter, max_patience=max_patience, add_noise=True, path=path)
        self.is_pretrained = True

        file = os.path.join(path, 'general_training_loss.json')
        with open(file, 'w') as f:
            json.dump(g_training_loss, f)
        print(f'Training loss saved to {file}')

        return g_training_loss

    def personalized_training(self, batch_size=32, target_update_period=100, iter=[400, 1500], max_patience=175, path='agent_state_finetuned'):
        assert self.is_pretrained == True, 'Agent must be pretrained before finetuning'

        ft_training_loss = self._train(batch_size=batch_size, target_update_period=target_update_period, iter=iter, max_patience=max_patience, add_noise=False, path=path)

        file = os.path.join(path, 'personalized_training_loss.json')
        with open(file, 'w') as f:
            json.dump(ft_training_loss, f)
        print(f'Training loss saved to {file}')

        return ft_training_loss

    def evaluate_policy(self, max_iter=int(4*24*60/3), render=False, print_output=True): # Max_iter is set to yield an episode for 4 days, given sample rate of 3 min
        CGM_idx = int(self.state_dim/3 - 1)
        CHO_idx = int(2*self.state_dim/3 - 1)

        # initialize metrics
        actor_output = []
        in_range = {'target': 0, 'hypo': 0, 'hyper': 0, 'total': 0}
        metrics = {'is_alive': True}

        state, info = self.env.reset()
        last_meal = 0
        for t in range(max_iter):
            
            if print_output:
                print(40*' ', end='\r')
                print('Still alive' + (t%4)*'.' + (4-(t%4))*' ' + f'({t}/{max_iter})', end='\r')
            
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
                if print_output:
                    metrics['is_alive'] = False
                    print(40*' ', end='\r')  
                    print(f'Episode finished after {t+1}/{max_iter} timesteps (patient died).')
                break

        if print_output:
            print(40*' ', end='\r')
            print('Episode finished.')

        metrics['actor_output'] = actor_output
        metrics['TIR'] = in_range['target']/in_range['total']
        metrics['hypo'] = in_range['hypo']/in_range['total']
        metrics['hyper'] = in_range['hyper']/in_range['total']

        return metrics

        
        