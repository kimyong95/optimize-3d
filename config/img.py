from config.base import get_config as get_base_config

def get_config():
    config = get_base_config()

    config.run_name = "img"

    # total objective evaluations: 25*16*4=1600
    config.batch_size = 16        # b_0 in the paper
    config.expansion_size = 4     # b_1 in the paper

    config.objectives = "scaled-drag-force;scaled-lift-force"

    return config
