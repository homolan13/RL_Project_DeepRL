from Agent import DDPGAgent
from utils import make_env

def main():

    average_patient = make_env('simglucose_adolescent_average', 'average_adolescent', print_space=False)
    ado1 = make_env('simglucose_ado_001', 'adolescent#001', print_space=False)
    agent = DDPGAgent(average_patient)

    _ = agent.general_training(path='adolescent_general_training_state')
    _ = agent.personalized_training(individual_env=ado1, path='adolescent_personalized_training_state')

if __name__ == '__main__':
    main()