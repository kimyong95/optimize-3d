import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.num_inference_steps = 50
    config.guidance_scale = 4.5
    config.objectives = "drag-coefficient"

    config.reward_server_port = 8000

    return config
