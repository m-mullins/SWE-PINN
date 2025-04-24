import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train_eval"   # "train" or "eval" or "train_eval"
    config.dataset = "burgers1d_Re1000.mat"
    config.nu = None
    
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Burgers"
    wandb.name = "pirate_rad"
    wandb.tag = ["ablation"]
    wandb.notes = "test"

    # Physics-informed initialization
    config.use_pi_init = True
    config.pi_init_type = "initial_condition"   # "linear_pde" or "initial_condition"

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )
    arch.pi_init = None # Leave as none, is updated with weights in train script

    # Transfer learning
    config.transfer = transfer = ml_collections.ConfigDict()
    transfer.curriculum = False  # Curriculum learning scheme
    transfer.curri_step = None    # Iteration from which init state will be passed for curriculum learning, None will take the last iter
    transfer.datasets = None
    transfer.s2s = False        # Sequence to sequence learning

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
    training.max_steps = 80000    # 80000
    training.batch_size_per_device = 2048   # 4096
    training.num_time_windows = 1
    training.use_lbfgs = False
    training.current_optim = None
    training.lbfgs_batch_size = 9216 # 20960, Must be dividable by 32 and have an round square root
    training.lbfgs_max_steps = 5000 # 5000
    training.lbfgs_c1 = 1e-4  # Armijo parameter (sufficient decrease) (1e-4)
    training.lbfgs_c2 = 0.9   # Curvature parameter (0.9)
    training.lbfgs_history_size = 10   # History size (10)
    training.grad_clip = 1e3    # Gradient clipping tolerance

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32
    weighting.use_rba = False   # Residual based attention mask for collocation points
    weighting.rba_gamma = 0.999 # RBA decay parameter
    weighting.rba_eta   = 0.01 # RBA learning rate parameter
    weighting.rba_sampler = "structured_random" # "fixed" or "structured_random"
    weighting.use_rad = True    # Collocation points sampling using Residual-based adaptive distribution
    weighting.rad_update_every_steps = 5
    weighting.rad_k = 1.0 # RAD sensitivity of the sampling PDF to the magnitude of the residuals
    weighting.rad_c = 1.0 # RAD stabilization constant to avoid extreme values and ensure numerical stability

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.global_step = None      # Leave as none, updated automatically in train with curriculum
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_nonlinearities = True
    logging.log_rba = False
    logging.log_rba_every_steps = [100,1000,5000,10000,20000,40000]
    logging.log_colloc_every_steps = [0,10,100,1000,5000,10000,20000,40000]

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 50000   # 10000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
