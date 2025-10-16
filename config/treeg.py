import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.num_inference_steps = 25
    config.guidance_scale = 7.0

    config.run_name = "treeg"
    
    config.ref_mesh_path = "drag-force/assets/sample.stl"
    config.objective = "drag-coefficient"

    # total objective evaluations: 25*16*4*1=1600
    config.total_num_samples = 16
    config.batch_size = 1 # active set size A in the paper
    config.expansion_size = 4 # branch out sample size K in the paper

    return config
