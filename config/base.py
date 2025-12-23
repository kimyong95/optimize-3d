import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.num_inference_steps = 25
    config.guidance_scale = 7.0
    config.objective = "drag-coefficient"

    config.reward_server_port = 8000

    return config
