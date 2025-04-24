# DETERMINISTIC
import os
import sys

# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags

# jax.config.update("jax_enable_x64", True)

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

config_flags.DEFINE_config_file(
    "config",
    "./configs/plain.py",
    # "./configs/default.py",
    # "./configs/pirate.py",
    # "./configs/pirate_rba.py",
    # "./configs/pirate_rad.py",
    # "./configs/sota_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv): 
    if FLAGS.config.mode == "train":        # Regular training
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":       # Regular evalutation of results
        eval.evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "train_eval": # Train and then evaluate the results
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
        eval.evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "curri":      # Implement curriculum learning scheme from Krishnapriyan et. al. (2021)
        datasets = FLAGS.config.transfer.datasets
        iterations = FLAGS.config.transfer.iterations
        FLAGS.config.logging.global_step = 0
        for dataset, iteration in zip(datasets, iterations):
            FLAGS.config.dataset = dataset  # Update the config to use the current dataset
            FLAGS.config.training.max_steps = iteration  # Update the config to use the current dataset
            logging.info(f"Training and evaluating with dataset: {dataset} for {iteration} iterations")
            train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
            eval.evaluate(FLAGS.config, FLAGS.workdir)
            FLAGS.config.logging.global_step = FLAGS.config.logging.global_step + FLAGS.config.training.max_steps

            FLAGS.config.use_pi_init = False # Remove pi init for following datasets
            FLAGS.config.transfer.s2s_pi_init = False   # Remove pi init for following datasets
            FLAGS.config.transfer.curri_step = iteration # Update iteration from which the next dataset will init it's params and state


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)