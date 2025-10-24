import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.epoches = 50
    config.num_inference_steps = 25
    config.guidance_scale = 7.0
    
    config.run_name = "grpo"
    
    # total objective evaluations: 32*50=1600
    config.training_samples_per_epoch = 32
    config.sampling_max_batch_size_per_device = 4
    config.training_max_batch_size_per_device = 1
    config.training_effective_batch_size = 16
    
    config.eval_max_batch_size_per_device = 16
    config.eval_samples = 16
    config.eval_freq = 10

    config.ref_mesh_path = "drag-force/assets/sample.stl"
    config.objective = "drag-coefficient"

    config.learning_rate = 1e-4

    config.clip_range = 0.1
    config.adv_clip_max = 5.0
    config.kl_beta = 0.04
    config.max_grad_norm = 1.0

    return config
