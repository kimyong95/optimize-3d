from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "dsearch"
    
    # algorithm-specific setting
    # total objective evaluation ≈ 76 * 25 = 1900
    # but actually due to rounding it is = 1626
    config.init_batch_size = 32      # b_0 in the paper
    config.final_batch_size = 16     # b_1 in the paper
    config.evaluation_budget = 76    # C in the paper

    return config
