from Agent import DDPGAgent
from utils import make_env

def main():

    adu1 = make_env('simglucose_adu_001', 'adult#001', print_space=False)
    agent = DDPGAgent(adu1)
    agent.is_pretrained = True # Manual override
    agent.load_agent('adult_general_training_state') # load the general training model

    _ = agent.personalized_training(path='adult_personalized_training_state', iter=[300, 1000])

if __name__ == '__main__':
    main()