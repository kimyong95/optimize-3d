import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.num_inference_steps = 25
    config.guidance_scale = 7.0

    config.run_name = "dsearch"
    
    config.ref_mesh_path = "drag-force/assets/sample.stl"
    config.objective = "drag-coefficient"

    # algorithm-specific setting
    # total objective evaluation ≈ 76 * 25 = 1900
    # but actually due to rounding it is ≈ 1626
    config.init_batch_size = 32      # b_0 in the paper
    config.final_batch_size = 16     # b_1 in the paper
    config.evaluation_budget = 76    # C in the paper

    config.reward_server_port = 8000

    return config
