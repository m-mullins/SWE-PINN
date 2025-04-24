import os
import sys

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
os.environ["WANDB__SERVICE_WAIT"] = "120"

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Add project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import train
import eval


FLAGS = flags.FLAGS

workdir = os.path.dirname(os.path.abspath(__file__))
flags.DEFINE_string("workdir", workdir, "Directory to store model data.")
# flags.DEFINE_string("workdir", os.path.join(workdir, "burgers"), "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    # "./configs/plain.py",
    # "./configs/default.py",
    # "./configs/sota.py",
    # "./configs/pirate.py",
    "./configs/pirate_rad.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    if FLAGS.config.mode == "train":        # Regular training
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":       # Regular evaltation of results
        eval.evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_s2s":       # Regular evaltation of results
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)

    # elif FLAGS.config.mode == "train_eval": # Train and then evaluate the results
    #     train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    #     eval.evaluate(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "train_eval": # Train and then evaluate the results
        train.train_and_evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "curri":      # Implement curriculum learning scheme from Krishnapriyan et. al. (2021)
        datasets = FLAGS.config.transfer.datasets
        FLAGS.config.logging.global_step = 0
        for dataset in datasets:
            FLAGS.config.dataset = dataset  # Update the config to use the current dataset
            logging.info(f"Training and evaluating with dataset: {dataset}")
            train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
            eval.evaluate(FLAGS.config, FLAGS.workdir)
            FLAGS.config.logging.global_step = FLAGS.config.logging.global_step + FLAGS.config.training.max_steps
    
if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
