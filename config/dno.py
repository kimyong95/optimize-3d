import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.prompt = "car"
    config.optimization_steps = 100
    config.num_inference_steps = 25
    config.guidance_scale = 7.0

    config.run_name = "dno"
    
    config.ref_mesh_path = "drag-force/assets/sample.stl"
    config.objective = "drag-coefficient"

    # total objective evaluations: 20*16*(4+1)=1600
    config.optimization_steps = 20
    config.total_num_samples = 16
    config.batch_size = 4 # number of samples for gradient estimation, it is equal to q in the paper

    return config
