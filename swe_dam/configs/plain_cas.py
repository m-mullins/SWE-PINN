import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train_eval"   # "train" or "eval" or "train_eval"
    # config.dataset = "swe_dam_2d_nx128_nt101_g0.1_x2.5.npz"
    # config.dataset = "swe_dam_2d_nx50_nt41_g0.1_x2.5.npz"
    config.dataset = "radial_dam_break_data_inter_t101_xy101_med.npz"
    # config.dataset = "radial_dam_break_data_inter_t41_xy51.npz"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-SWE-DAM"
    wandb.name = "plain_cas"
    wandb.tag = None
    wandb.notes = "impact study"

    # Problem setup
    config.setup = setup = ml_collections.ConfigDict()
    setup.g = 9.8           # Gravity constant
    setup.manning = 0       # Manning coefficient
    setup.use_visc = False   # Use numerical viscosity in residual eqn

    # Nondimensionalization
    config.nondim = nondim = ml_collections.ConfigDict()
    nondim.nondimensionalize = True
    nondim.U_star = 5.0     # Velocity sqrt(gH)
    nondim.L_star = 40.0    # Length
    nondim.H_star = 2.5     # Height
    nondim.T_star = None    # Time (calculated from U and L in train.py)
    nondim.Froude = None    # Caracteristic Froude number (calculated in train.py)
    nondim.visc = 0.0001    # Caracteristic numerical viscosity (calculated in train.py)

    # Physics-informed initialization
    config.use_pi_init = False
    config.pi_init_type = "initial_condition"   # "initial_condition"

    # Transfer learning
    config.transfer = transfer = ml_collections.ConfigDict()
    transfer.curriculum = False     # Curriculum learning scheme
    transfer.datasets = ["swe_dam_2d_nx50_nt41_g0.1.npz","swe_dam_2d_nx50_nt41_g1.0.npz"]  # List of dataset filenames for curriculum training
    transfer.iterations = [10, 20]  # List of training iterations for each dataset
    transfer.curri_step = None      # Leave as none. Iteration from which init state will be passed for curriculum learning
    transfer.s2s_transfer = True    # Use transfer learning to initiate params of subsequent time windows in seq-2-seq learning
    transfer.s2s_pi_init = True     # Leave as True if s2s_transfer is also True. Will change to false after first window.

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 8
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "gelu"
    arch.periodicity = False
    arch.fourier_emb = None
    arch.reparam = None


    # arch.nonlinearity = 0.0 # alpha
    arch.pi_init = None # Leave as none, is updated with weights in train script

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
    training.max_steps = 120000             # 20000
    training.batch_size_per_device = 2048   # 2048
    training.s2s = True                     # Sequence to sequence learning
    training.num_time_windows = 1           # For seq2seq
    training.g_schedule = None              # Increase g during training according to a schedule "step", "sigmoid" or None
    training.g_min = 1.0                    # Min value of g for the schedule
    training.ratio = 1                      # Downsampling ratio for l2 eval ratio during training
    training.grad_clip = 1e3                # Gradient clipping tolerance

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = {
        "u_ic": 1.0,    # ic loss
        "v_ic": 1.0,    # ic loss
        "h_ic": 1.0,    # ic loss
        # "slip_bc": 1.0,    # bc loss
        # "outflow_bc": 1.0,    # bc loss
        # "h_bc": 1.0,    # bc loss
        "ru": 1.0,      # res loss (momentum)
        "rv": 1.0,      # res loss (momentum)
        "rc": 1.0,      # res loss (continuity)
    }

    weighting.momentum = 0.9
    weighting.update_every_steps = 100          # 100 for grad norm and 1000 for ntk
    weighting.update_every_steps_lbfgs = 5000   # 100 for grad norm and 1000 for ntk
    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32                   # 32
    weighting.use_rba = False                   # Residual based attention mask for collocation points
    weighting.rba_gamma = 0.999                 # RBA decay parameter
    weighting.rba_eta   = 0.01                  # RBA learning rate parameter
    weighting.rba_sampler = "fixed"             # "fixed" or "structured_random"
    weighting.use_rad = False                   # Collocation points sampling using Residual-based adaptive distribution
    weighting.rad_update_every_steps = 5
    weighting.rad_k = 1.0                       # RAD sensitivity of the sampling PDF to the magnitude of the residuals, higher = more focused on high residual zones
    weighting.rad_c = 1.0                       # RAD stabilization constant to avoid extreme values, higher = more randomness

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.log_every_steps_lbfgs = 500
    logging.global_step = None                  # Leave as none, updated automatically in train with curriculum
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False
    logging.log_nonlinearities = False
    logging.log_rba = False
    logging.log_rba_every_steps = [100,1000,5000,10000,20000,40000]
    logging.log_colloc_every_steps = [0,10,100,1000,5000,10000,20000,40000]

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 50000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config