from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "dsearch"
    
    # algorithm-specific setting
    # target objective evaluation 32 * 50 = 1600
    # but actually due to rounding it is = 1576
    config.init_batch_size = 8      # b_0 in the paper
    config.final_batch_size = 4     # b_1 in the paper
    config.evaluation_budget = 32    # C in the paper

    return config
