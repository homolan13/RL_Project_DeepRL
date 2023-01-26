from Agent import DDPGAgent
from utils import make_env
import json
import os
from tqdm import tqdm

def evaluation(agent, envs, json_path, personalized):
    metrics = []
    for i in tqdm(range(10)):
        if not personalized:
            agent.change_env(envs[i])
        m = agent.evaluate_policy()
        metrics.append(m)

    with open(json_path, 'w') as f:
        json.dump(metrics, f)

def ado_g():
    print('Adolescent General Evaluation')
    average = make_env('simglucose_average', 'average_adolescent')
    agent = DDPGAgent(average)
    agent.load_agent('adolescent_general_training_state') # load the general training model
    envs = [make_env(f'simglucose_ado_0{i:02d}', f'adolescent#0{i:02d}', print_space=False) for i in range(1,11)]

    evaluation(agent, envs, os.path.join('metrics','ado_g_metrics.json'), False)

def adu_g():
    print('Adult General Evaluation')
    average = make_env('simglucose_average', 'average_adult')
    agent = DDPGAgent(average)
    agent.load_agent('adult_general_training_state') # load the general training model
    envs = [make_env(f'simglucose_adu_0{i:02d}', f'adult#0{i:02d}', print_space=False) for i in range(1,11)]

    evaluation(agent, envs, os.path.join('metrics','adu_g_metrics.json'), False)

def ado_p():
    print('Adolescent Personalized Evaluation')
    average = make_env('simglucose_average', 'average_adolescent')
    agent = DDPGAgent(average)
    agent.load_agent('adolescent_personalized_training_state') # load the general training model
    env = make_env(f'simglucose_ado_001', f'adolescent#001', print_space=False)

    agent.change_env(env)
    evaluation(agent, os.path.join('metrics','ado_p_metrics.json'), True)

def adu_p():
    print('Adult Personalized Evaluation')
    average = make_env('simglucose_average', 'average_adult')
    agent = DDPGAgent(average)
    agent.load_agent('adult_personalized_training_state') # load the general training model
    env = make_env(f'simglucose_adu_001', f'adult#001', print_space=False)

    agent.change_env(env)
    evaluation(agent, os.path.join('metrics','adu_p_metrics.json'), True)

    

if __name__ == '__main__':
    ado_g()
    adu_g()
    ado_p()
    adu_p()