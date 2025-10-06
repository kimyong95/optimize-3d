import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.optimization_steps = 1000
    config.image_path = "TripoSG/assets/example_data/car.png"
    config.num_inference_steps = 25
    config.guidance_scale = 7.0

    config.run_name = "optimize-3d"
    config.batch_size = 8
    
    config.eval_freq = 1
    
    config.ref_mesh_path = "drag-force/assets/drivaer_1_single_solid.stl"
    config.objective = "drag-coefficient"
    
    config.optimize_slat = True

    return config
