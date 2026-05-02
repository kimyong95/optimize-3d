from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "evolvable-3d"

    # total objective evaluations: total_num_samples * perturbation_samples * num_inference_steps
    # e.g. 32 * 50 = 1600
    config.perturbation_samples = 32 # N: noise directions sampled per step (K in ES literature)

    return config
