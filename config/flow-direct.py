from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.optimization_steps = 100

    config.run_name = "flow-direct"
    config.total_num_samples = 16
    config.max_batch_size_per_device = 16

    return config
