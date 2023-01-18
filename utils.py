import simglucose
import gym
from gym.wrappers import FlattenObservation

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
    elif 300 < BG and BG <= 350:
        return -1
    # Other cases
    else:
        return -2

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