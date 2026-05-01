from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "evolvable-3d"

    # total objective evaluations: total_num_samples * perturbation_samples * num_inference_steps
    # e.g. 64 * 25 = 1600
    config.perturbation_samples = 64 # N: noise directions sampled per step (K in ES literature)
    config.noise_level = 0.7         # SDE noise scale (same default as treeg)

    return config
