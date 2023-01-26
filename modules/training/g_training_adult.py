from Agent import DDPGAgent
from utils import make_env

def main():

    average_patient = make_env('simglucose_adult_average', 'average_adult', print_space=False)
    agent = DDPGAgent(average_patient)

    _ = agent.general_training(path='adult_general_training_state', iter=[600, 1500])

if __name__ == '__main__':
    main()