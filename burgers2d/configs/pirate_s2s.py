import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train_eval_s2s"   # "train" or "eval" or "train_eval" or "eval_s2s" or "train_eval_s2s"
    config.dataset = "burgers2d_nt101_nx101_ny101_Re237.mat"
    config.nu = None        # Leave as none, updated in train script
    config.reynolds = None  # Leave as none, updated in train script
    
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Burgers2d"
    wandb.name = "pirate_s2s"
    wandb.tag = ["s2s"]

    # Nondimensionalization
    config.nondim = False

    # Physics-informed initialization
    config.use_pi_init = True
    config.pi_init_type = "initial_condition"   # "linear_pde" or "initial_condition"

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 2
    arch.activation = "swish" # "tanh"
    arch.periodicity = None
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}     # RWF
    )
    arch.nonlinearity = 0.0 # alpha
    arch.pi_init = None # Leave as none, is updated with weights in train script

    # Transfer learning
    config.transfer = transfer = ml_collections.ConfigDict()
    transfer.curriculum = False # Curriculum learning scheme
    transfer.datasets = None    # List of dataset filenames for curriculum training
    transfer.iterations = None  # List of training iterations for each dataset
    transfer.curri_step = None  # Leave as none. Iteration from which init state will be passed for curriculum learning

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.staircase = False
    optim.warmup_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 30000 # 30000
    training.batch_size_per_device = 2048 # 2048
    training.s2s = True                 # Sequence to sequence learning
    training.num_time_windows = 3       # For seq2seq

    # Weighting of loss terms
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"  # "grad_norm" or "ntk"
    weighting.init_weights = ml_collections.ConfigDict({"u_ic": 10.0, "v_ic": 10.0, "u_bc": 1.0, "v_bc": 1.0, "ru": 1.0, "rv": 1.0}) # lambda
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True # Respecting Temporal Causality algorithm
    weighting.causal_tol = 1.0  # epsilon
    weighting.num_chunks = 32   # number of subdivisions

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500   # 500
    logging.global_step = None      # Leave as none, updated automatically in train with curriculum
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000 # 10000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models (amount of dim in domain)
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
