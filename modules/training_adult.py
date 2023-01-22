from Agent import DDPGAgent
from utils import make_env

def main():

    average_patient = make_env('simglucose_adult_average', 'average_adult', print_space=False)
    adu1 = make_env('simglucose_adult_001', 'adult#001', print_space=False)
    agent = DDPGAgent(average_patient)

    _ = agent.general_training(path='adult_general_training_state')
    _ = agent.personalized_training(individual_env=adu1, path='adult_personalized_training_state')

if __name__ == '__main__':
    main()