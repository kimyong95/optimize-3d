from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "dno"
    
    # total objective evaluations: 20*16*(4+1)=1600
    config.optimization_steps = 20
    config.total_num_samples = 16
    config.batch_size = 4 # number of samples for gradient estimation, it is equal to q in the paper

    return config
