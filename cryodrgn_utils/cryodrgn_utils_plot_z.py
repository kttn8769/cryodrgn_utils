'''Plot cryoDRGN z.pkl files
'''

import sys
import os
import re
import pickle
import argparse

import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


DEFAULT_MARKER_SIZE = int(mpl.rcParams['lines.markersize'] ** 2)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        '--infiles', nargs='+', help='Input z.*.pkl files, where "*" represents the epoch number.'
    )
    parser.add_argument(
        '--outdir', help='Output directory path. If not specified, output to the current directory.'
    )
    parser.add_argument(
        '--generate-movie', action='store_true',
        help='Combine generated figures into a single movie file (.gif)'
    )
    parser.add_argument(
        '--figsize', type=int, default=5, help='Figure size in inches.'
    )
    parser.add_argument(
        '--dpi', type=int, default=100, help='Figure dpi.'
    )
    parser.add_argument(
        '--marker-size', type=int, default=DEFAULT_MARKER_SIZE, help='Marker size in scatter plot.'
    )
    parser.add_argument(
        '--outfile-prefix', default='z', help='Output filename prefix.'
    )
    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main():
    args = parse_args()

    infiles = []
    epochs = []
    for infile in args.infiles:
        assert os.path.exists(infile), 'File not exists.'
        infiles.append(infile)

        m = re.match(r'z.([0-9]+).pkl', os.path.basename(infile))
        assert m is not None, 'Cannot parse epoch number from filename.'
        epoch = int(m.group(1))
        epochs.append(epoch)

    infiles = np.array(infiles)
    epochs = np.array(epochs)

    idxs_sort = np.argsort(epochs)
    infiles_sorted = infiles[idxs_sort]
    epochs_sorted = epochs[idxs_sort]

    # PCA with the final epoch's data
    pca = PCA(n_components=2)
    with open(infiles_sorted[-1], 'rb') as f:
        Z = pickle.load(f)
    assert Z.shape[1] >= 2, 'For this program, latent vector dimensions must be larger than or equal to 2.'
    pca.fit(Z)

    Zs = []
    for infile in infiles_sorted:
        with open(infile, 'rb') as f:
            Z = pickle.load(f)
        Z_proj = pca.transform(Z)
        Zs.append(Z_proj)
    Zs = np.array(Zs)

    xmin, xmax = np.min(Zs[:, :, 0]), np.max(Zs[:, :, 0])
    ymin, ymax = np.min(Zs[:, :, 1]), np.max(Zs[:, :, 1])

    if args.outdir is None:
        outdir = os.getcwd()
    else:
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)

    output_figs = []
    for Z, epoch in zip(Zs, epochs_sorted):
        fig, ax = plt.subplots(figsize=(args.figsize, args.figsize), dpi=args.dpi)
        ax.scatter(Z[:, 0], Z[:, 1], s=args.marker_size)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(f'Epoch {epoch:03d}')
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        fig.set_tight_layout(True)
        output_fig = os.path.join(outdir, f'{args.outfile_prefix}_epoch_{epoch:03d}.png')
        fig.savefig(output_fig, facecolor='white')
        output_figs.append(output_fig)
        plt.close(fig)

    if args.generate_movie:
        output_mov = os.path.join(outdir, f'{args.outfile_prefix}_movie.gif')
        frames = []
        for output_fig in output_figs:
            frames.append(imageio.imread(output_fig))
        imageio.mimsave(output_mov, frames)
