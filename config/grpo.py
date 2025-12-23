from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.epoches = 50
    
    config.run_name = "grpo"
    
    # total objective evaluations: 32*50=1600
    config.training_samples_per_epoch = 32
    config.gradient_updates_per_epoch = 2
    
    config.eval_samples = 16
    config.eval_freq = 10

    config.sampling_max_batch_size_per_device = 4
    config.training_max_batch_size_per_device = 1
    config.eval_max_batch_size_per_device = 16

    config.learning_rate = 1e-4
    config.clip_range = 0.1
    config.adv_clip_max = 5.0
    config.kl_beta = 0.04
    config.max_grad_norm = 1.0

    return config
