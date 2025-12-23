from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "treeg"
    
    # total objective evaluations: 25*16*4*1=1600
    config.total_num_samples = 16
    config.batch_size = 1 # active set size A in the paper
    config.expansion_size = 4 # branch out sample size K in the paper

    return config
