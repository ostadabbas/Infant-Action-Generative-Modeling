import os

import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from .visualize import viz_generated_motions
import numpy as np
import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()

    motions = np.load(os.path.join(folder, 'synthetic_samples.npy'), allow_pickle=True).item()

    # visualize_params
    viz_generated_motions(motions, parameters, folder=folder)


if __name__ == '__main__':
    main()
