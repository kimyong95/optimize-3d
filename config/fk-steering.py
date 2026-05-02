from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "fk-steering"

    # total objective evaluations: batch_size * num_inference_steps
    # e.g. 32 * 50 = 1600
    config.batch_size = 32       # N: number of FK particles
    config.beta = 10.0           # FK steering inverse temperature
    config.noise_level = 0.7     # SDE noise scale (same default as treeg)

    return config
