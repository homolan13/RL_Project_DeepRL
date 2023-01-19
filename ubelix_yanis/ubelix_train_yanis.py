from Agent import DDPGAgent
from utils import make_env

def main():
    average_patient = make_env('simglucose_average', 'average_adolescent', print_space=False)
    agent = DDPGAgent(average_patient)

    _ = agent.general_training()

if __name__ == 'main':
    main()