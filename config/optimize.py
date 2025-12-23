from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.optimization_steps = 100
    config.noise_level = 0.7

    config.run_name = "optimize-3d"
    config.total_num_samples = 16
    config.max_batch_size_per_device = 16
    
    config.eval_freq = 10
    
    config.lr = 1.0

    return config
