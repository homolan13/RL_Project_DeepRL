from Agent import DDPGAgent
from utils import make_env

def main():

    ado1 = make_env('simglucose_ado_001', 'adolescent#001', print_space=False)
    agent = DDPGAgent(ado1)
    agent.is_pretrained = True # Manual override
    agent.load_agent('adolescent_general_training_state') # load the general training model

    _ = agent.personalized_training(path='adolescent_personalized_training_state')

if __name__ == '__main__':
    main()