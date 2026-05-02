from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "treeg"
    
    # total objective evaluations: 50*8*4=1600
    config.batch_size = 4 # active set size A in the paper
    config.expansion_size = 8 # branch out sample size K in the paper

    return config
