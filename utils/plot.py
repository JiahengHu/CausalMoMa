import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import to_numpy


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


# ---------------------------------- TO CREATE A SERIES OF PICTURES ---------------------------------- #
# from https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/

def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = '%s%03d.jpeg' % (prefix, i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files


# ----------------------- TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION ----------------------- #

def make_movie(files, output, fps=10, bitrate=1800, **kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = {'.mp4': 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                       % (",".join(files), fps, output_name, bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s' % (output_name, fps, output)

    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])


def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s' % (delay, loop, " ".join(files), output))


def make_strip(files, output, **kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s' % (" ".join(files), output))


# ---------------------------------------------- MAIN FUNCTION ---------------------------------------------- #

def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.mp4': make_movie,
         '.ogv': make_movie,
         '.gif': make_gif,
         '.jpeg': make_strip,
         '.png': make_strip}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)


def plot_adjacency_intervention_mask(model, writer, step, adjacency=None):
    if adjacency is None:
        adjacency = model.get_adjacency()
        if adjacency is None:
            return
    adjacency_intervention = to_numpy(adjacency)

    feature_dim, action_dim = adjacency_intervention.shape

    fig = plt.figure(figsize=(action_dim * 0.45 + 2, feature_dim * 0.45 + 2))

    vmax = adjacency[0, -1]
    while vmax < 0.1:
        vmax = vmax * 10
        adjacency_intervention = adjacency_intervention * 10
    sns.heatmap(adjacency_intervention, linewidths=3, vmin=0, vmax=vmax, square=True, annot=True, fmt='.2f', cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

    fig.tight_layout()
    if writer:
        writer.add_figure("adjacency", fig, step + 1)
    else:
        plt.show()
    plt.close("all")


def plot_adjacency(adjacency):

    adjacency_intervention = to_numpy(adjacency)

    feature_dim, action_dim = adjacency_intervention.shape

    fig = plt.figure(figsize=(action_dim * 0.45 + 2, feature_dim * 0.45 + 2))

    vmax = 1
    sns.heatmap(adjacency_intervention, linewidths=3, vmin=0, vmax=vmax, square=True, annot=True, cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

    # This is to plot the axis caption
    n_action = adjacency.shape[0]
    n_reward = adjacency.shape[-1]

    plt.xticks(np.array(range(n_reward)) + 0.5, ["R_up/down", "R_left/right", "R_3", "R_4", "R_5"])
    plt.yticks(np.array(range(n_action)) + 0.5, ["A", "A2", "A3", "A4"], rotation=90)

    fig.tight_layout()

    plt.show()
    plt.close("all")


if __name__ == '__main__':
    adjacency =  torch.tensor([ [     0.871,      0.000,      0.000,      0.000, 0.02],
                                [     0.000,      0.868,      0.000,      0.000, 0.02],
                                [     0.040,      0.041,      0.000,     -0.000, 0.02],
                                [     0.000,      0.000,      0.034,      0.000, 0.02],
                                [    -0.000,      0.000,      0.000,      0.028, 0.02]])
    # |omni  ,|head,|      arm    ,|gr
    # adjacency =  torch.tensor([[1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # Reach
    # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # EE Local Orientation
    # [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # EE Local Position
    # [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Base Collision
    # [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # Arm Collision
    # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # Self Collision
    # [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # Head Attention
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],])  # Gripper Grasp

    # adjacency =  torch.tensor([ [     0.871,      0.000,      0.000,      0.000],
    #                             [     0.000,      0.868,      0.000,      0.000],
    #                             [     0.040,      0.041,      0.000,     -0.000],
    #                             [     0.000,      0.000,      0.034,      0.000],
    #                             [    -0.000,      0.000,      0.000,      0.028]])
    adjacency =  torch.tensor([ [     1,      0,      0,      0],
                                [     0,      1,      0,      0],
                                [     1,      1,      0,     0],
                                [     0,      0,      1,      0],
                                [    0,      0,      0,      1]]).T
    adjacency[0, -1] = 1

    # plot_adjacency(adjacency)
    plot_adjacency_intervention_mask(None, None, None, adjacency=adjacency)