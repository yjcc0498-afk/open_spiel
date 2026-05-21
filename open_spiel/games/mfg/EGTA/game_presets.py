"""Game-specific hyperparameter presets aligned with the paper's Table 4."""


TABLE4_PRESETS = {
    # 10 states, horizon 10, 10 strategies, ~3000 samples.
    "mean_field_lin_quad": {
        "display_name": "Linear Quadratic",
        "state_count": 10,
        "game_size": 10,
        "game_horizon": 10,
        "num_strategies": 10,
        "egta_iterations": 9,
        "num_samples": 3000,
        "sampling_mode": "hybrid",
    },
    # 100 states on a 1D line, horizon 30, 18 strategies, ~6000 samples.
    "mfg_crowd_modelling": {
        "display_name": "1-D Crowd",
        "state_count": 100,
        "game_size": 100,
        "game_horizon": 30,
        "num_strategies": 18,
        "egta_iterations": 17,
        "num_samples": 6000,
        "sampling_mode": "hybrid",
    },
    # 100 states total = 10x10 grid, horizon 30, 18 strategies, ~6000 samples.
    "mfg_crowd_modelling_2d": {
        "display_name": "2-D Crowd",
        "state_count": 100,
        "game_size": 10,
        "game_horizon": 30,
        "num_strategies": 18,
        "egta_iterations": 17,
        "num_samples": 6000,
        "sampling_mode": "hybrid",
    },
}


def apply_table4_preset(game_name, flags_obj, verbose=True):
    """Mutate known flags so listed games follow the paper preset automatically."""
    preset = TABLE4_PRESETS.get(game_name)
    if preset is None:
        return None

    def _was_explicitly_set(name):
        try:
            return flags_obj[name].present > 0
        except (KeyError, TypeError, AttributeError):
            return False

    def _set_default(name, value):
        if hasattr(flags_obj, name) and not _was_explicitly_set(name):
            setattr(flags_obj, name, value)

    _set_default("game_size", preset["game_size"])
    _set_default("game_horizon", preset["game_horizon"])
    _set_default("egta_iterations", preset["egta_iterations"])
    _set_default("num_samples", preset["num_samples"])
    _set_default("sampling_mode", preset["sampling_mode"])
    _set_default("grid_density", 4)
    _set_default("grid_sample_count", 0)
    _set_default("dirichlet_sample_count", 0)

    if verbose:
        print(
            "Applied Table 4 preset for {}: states={}, size={}, horizon={}, strategies={}, egta_iterations={}, samples={}, sampling_mode={}".format(
                preset["display_name"],
                preset["state_count"],
                getattr(flags_obj, "game_size", preset["game_size"]),
                getattr(flags_obj, "game_horizon", preset["game_horizon"]),
                preset["num_strategies"],
                getattr(flags_obj, "egta_iterations", preset["egta_iterations"]),
                getattr(flags_obj, "num_samples", preset["num_samples"]),
                getattr(flags_obj, "sampling_mode", preset["sampling_mode"]),
            )
        )
    return preset
