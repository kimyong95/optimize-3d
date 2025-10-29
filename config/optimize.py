import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.optimization_steps = 100
    config.num_inference_steps = 25
    config.guidance_scale = 7.0

    config.run_name = "optimize-3d"
    config.batch_size = 16
    
    config.eval_freq = 10
    
    config.ref_mesh_path = "drag-force/assets/sample.stl"
    config.objective = "drag-coefficient"

    config.reward_server_port = 8000

    config.lr = 1.0

    return config
