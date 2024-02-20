import gc
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from matplotlib.font_manager import FontProperties

from utils.video_utils import frame_generator, figure_to_array, get_heatmap, get_heatmap_from_folder

# from trajectories.data_reading import get_candidates, get_heatmap, get_video_source
from trajectories.fitting import fit_trajectories
from trajectories.filtering import (build_path_mapping, build_trajectory_graph,
                                    find_next_nodes, find_prev_nodes,
                                    find_shortest_paths)


def annotate_detections(frame, predicted_positions, true_position=None, max_heatmap_value=None):
    """Put dots on the predicted (red) and true (green) ball positions.

    Parameters
    ----------
    frame : array
        frame to annotate
    predicted_positions : tuple of float
        predicted ball position in pixel coordinates. The coordinate order is `(x, y)`
        can be a list of tuples for multiple detections
    true_position : tuple of float, optional
        true ball position in pixel coordinates. The coordinate order is `(x, y)`
    max_heatmap_value : float or list of float, optional
        the value of the heatmap maximum.
        Can be a list for each detected position.
        If a single value is provided, will be annotated in the upper right corner.
        If a list is provided, will be annotated above each detection.

    Returns
    -------
    annotated_frame : array
        it has the same shape as `frame`.
    """
    annotated_frame = frame.copy()

    if true_position is not None:
        annotated_frame = cv2.circle(annotated_frame,
                                     center=true_position,
                                     radius=5,
                                     color=(0, 255, 0),
                                     thickness=cv2.FILLED)

    if len(np.shape(predicted_positions)) == 1:
        predicted_positions = (predicted_positions, )

    for predicted_position in predicted_positions:
        annotated_frame = cv2.circle(annotated_frame,
                                     center=predicted_position,
                                     radius=5,
                                     color=(255, 0, 0),
                                     thickness=cv2.FILLED)

    annotated_frame = cv2.addWeighted(annotated_frame, 0.6, frame, 0.4, 0)

    if max_heatmap_value is not None:
        if not hasattr(max_heatmap_value, '__iter__'):
            annotated_frame = cv2.putText(annotated_frame,
                                          text=f"{max_heatmap_value:.2g}",
                                          org=(int(0.85*frame.shape[1]), int(0.15*frame.shape[0])),
                                          fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                          fontScale=1,
                                          color=(255, 255, 255))
        else:
            for pos, value in zip(predicted_positions, max_heatmap_value):
                annotated_frame = cv2.putText(annotated_frame,
                                              text=f"{value:.2g}",
                                              org=(pos[0], pos[1] + 20),
                                              fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                              fontScale=0.6,
                                              color=(255, 255, 255))

    return annotated_frame


def get_stats_table(frame_index: int,
                    trajectory_info: dict,
                    s: int = 7):

    ann = f"{frame_index}".ljust(s+2) if frame_index is not None else " "*(s+2)
    ann += f"x" + " "*(s-1) + "y" + " "*(s-3) + "|.|"
    ann += "\n" + "\u2014"*(s*3+3) + "\n"

    if trajectory_info['found_trajectory']:
        p0 = trajectory_info['p0']
        v0 = trajectory_info['v']
        a = trajectory_info['a']

        if frame_index is not None:
            t = frame_index - trajectory_info['k_min']
            v = v0 + a*t
            p = p0 + v0*t + a*t*t/2
            ann += "p  " + f"{p[1]:.0f}".rjust(s) + f"{p[0]:.0f}".rjust(s)
            ann += "\n"
            ann += "v  " + f"{v[1]:.2f}".rjust(s) + f"{v[0]:.2f}".rjust(s) + f"{np.linalg.norm(v):.2f}".rjust(s)
            ann += "\n"
        ann += "p0 " + f"{p0[1]:.0f}".rjust(s) + f"{p0[0]:.0f}".rjust(s)
        ann += "\n"
        ann += "v0 " + f"{v0[1]:.2f}".rjust(s) + f"{v0[0]:.2f}".rjust(s) + f"{np.linalg.norm(v0):.2f}".rjust(s)
        ann += "\n"
        ann += "a  " + f"{a[1]:.2f}".rjust(s) + f"{a[0]:.2f}".rjust(s) + f"{np.linalg.norm(a):.2f}".rjust(s)
    else:
        if frame_index is not None:
            ann += "p  " + f"---".rjust(s) + f"---".rjust(s)
            ann += "\n"
            ann += "v  " + f"---".rjust(s) + f"---".rjust(s) + f"---".rjust(s)
            ann += "\n"
        ann += "p0 " + f"---".rjust(s) + f"---".rjust(s)
        ann += "\n"
        ann += "v0 " + f"---".rjust(s) + f"---".rjust(s) + f"---".rjust(s)
        ann += "\n"
        ann += "a  " + f"---".rjust(s) + f"---".rjust(s) + f"---".rjust(s)
    return ann


def display_frame_labels(ax,
                         display: str,
                         candidates: np.ndarray,
                         trajectory_info: dict,
                         starting_frame: int,
                         annotate: bool,
                         show_fitting_points: bool,
                         trajectory_color: str,
                         fontsize: int):
    k_seed = trajectory_info['k_seed']
    i_seed = trajectory_info['i_seed']

    k_min = trajectory_info['k_min']
    i_min = trajectory_info['i_min']

    k_mid = trajectory_info['k_mid']
    i_mid = trajectory_info['i_mid']

    k_max = trajectory_info['k_max']
    i_max = trajectory_info['i_max']

    sf = starting_frame

    bbox = {'boxstyle': 'round',
            'facecolor': trajectory_color,
            'edgecolor': 'none',
            'alpha': 0.4}
    font = FontProperties(family='monospace', weight='bold', size=fontsize)

    if 'all' in display:
        display = ['k_min', 'k_mid', 'k_max', 'k_seed']

    if 'k_seed' in display:
        if annotate:
            ax.annotate(k_seed, [candidates[k_seed-sf, i_seed, 1], candidates[k_seed-sf, i_seed, 0]], fontproperties=font, bbox=bbox, color='k')
        if show_fitting_points:
            ax.scatter(candidates[k_seed-sf, i_seed, 1], candidates[k_seed-sf, i_seed, 0], c='k', s=5)

    if 'k_min' in display:
        if annotate:
            ax.annotate(k_min, [candidates[k_min-sf, i_min, 1], candidates[k_min-sf, i_min, 0]], fontproperties=font, bbox=bbox, color='w')
        if show_fitting_points:
            ax.scatter(candidates[k_min-sf, i_min, 1], candidates[k_min-sf, i_min, 0], c='w', marker='^')

    if 'k_mid' in display:
        if annotate:
            ax.annotate(k_mid, [candidates[k_mid-sf, i_mid, 1], candidates[k_mid-sf, i_mid, 0]], fontproperties=font, bbox=bbox, color='w')
        if show_fitting_points:
            ax.scatter(candidates[k_mid-sf, i_mid, 1], candidates[k_mid-sf, i_mid, 0], c='w')

    if 'k_max' in display:
        if annotate:
            ax.annotate(k_max, [candidates[k_max-sf, i_max, 1], candidates[k_max-sf, i_max, 0]], fontproperties=font, bbox=bbox, color='w')
        if show_fitting_points:
            ax.scatter(candidates[k_max-sf, i_max, 1], candidates[k_max-sf, i_max, 0], c='w', marker='s')


def show_single_trajectory(fitting_info,
                           candidates,
                           k_seed,
                           frame=None,
                           heatmap=None,
                           k_min = None,
                           i_min = None,
                           k_max = None,
                           i_max = None,
                           ax=None,
                           show_outside_range=False,
                           display='k_min stats',
                           frame_index = None,
                           annotate=True,
                           show_fitting_points=False,
                           trajectory_color='y',
                           fontsize=11,
                           dpi=100,
                           alpha=0.8,
                           line_style='.-',
                           verbose=True,
                           dark_mode=True,
                           **kwargs):

    if ax is None:
        w, h = 1280, 720
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.figure

    # display frame
    if frame is not None:
        ax.imshow(frame, zorder=-200)
        if heatmap is not None:
            ax.imshow(cv2.resize(heatmap, (frame.shape[1], frame.shape[0])), zorder=-199, alpha=0.5, cmap='gray', vmin=0, vmax=1)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)

    # get trajectory_info and starting frame
    starting_frame = fitting_info['trajectories'][0]['k_seed']
    if k_seed is None:
        k_seed = starting_frame
    trajectory_info = fitting_info['trajectories'][k_seed-starting_frame]

    # display table of trajectory statistics
    if annotate and display is not None and ('parameters' in display or
                                             'params' in display or
                                             'p' in display or
                                             's' in display or
                                             'stats' in display or
                                             'all' in display):
        ann = get_stats_table(frame_index, trajectory_info)

        facecolor = '#232323'if dark_mode else '#FFFFFF'
        textcolor = '#FFFFFF' if dark_mode else '#000000'
        alpha = 0.85 if dark_mode else 0.7
        bbox = {'boxstyle': 'square,pad=0.5',
                'facecolor': facecolor,
                'edgecolor': textcolor,
                'linewidth': 0.5,
                'alpha': alpha}
        font = FontProperties(family='monospace', size=fontsize)
        ax.annotate(ann, [40, 40], fontproperties=font, bbox=bbox, color=textcolor, va='top')

    # if the trajectory is not found, return the axes as they are
    if not trajectory_info['found_trajectory']:
        if verbose:
            print(f'No fitted trajectory for frame {k_seed}')
        return ax

    # find k_min and and k_max
    if k_min is None:
        k_min = trajectory_info['k_min']
        i_min = trajectory_info['i_min']
    elif i_min is None:
        raise ValueError('You must pass both k_min and i_min')

    if k_max is None:
        k_max = trajectory_info['k_max']
        i_max = trajectory_info['i_max']
    elif i_max is None:
        raise ValueError('You must pass both k_max and i_max')

    # display trajectory
    trajectory = trajectory_info['trajectory']
    k = np.arange(len(trajectory)) + k_seed - starting_frame - (len(trajectory)-1)//2

    # offset k_min and k_max them by the starting frame to correctly index candidates
    k_min -= starting_frame
    k_max -= starting_frame
    if show_outside_range:
        ax.plot(trajectory[k<=k_min,1], trajectory[k<=k_min,0], line_style, color=trajectory_color, zorder=-1, alpha=alpha/4, **kwargs)
        ax.plot(trajectory[k>=k_max,1], trajectory[k>=k_max,0], line_style, color=trajectory_color, zorder=-1, alpha=alpha/4, **kwargs)
    mask = np.logical_and(k>=k_min, k<=k_max)
    ax.plot(trajectory[mask,1], trajectory[mask,0], line_style, color=trajectory_color, zorder=-1, alpha=alpha, **kwargs)

    # Display frame labels
    if display is not None:
        display_frame_labels(ax,
                             display,
                             candidates,
                             trajectory_info,
                             starting_frame,
                             annotate,
                             show_fitting_points,
                             trajectory_color,
                             fontsize)

    ax.set_axis_off()
    if frame is not None:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return ax


def show_trajectory_sequence(fitting_info: dict,
                             candidates: np.ndarray,
                             sequence: list,
                             frame: np.ndarray = None,
                             heatmap: np.ndarray = None,
                             ax=None,
                             dpi=100,
                             link_trajectories=True,
                             colors=None,
                             **kwargs):
    """Show trajectories superimposed on one frame.

    Parameters
    ----------
    fitting_info : dict
        _description_
    candidates : np.ndarray
        detection candidates
    sequence : list of int
        trajectory sequence. Usually it is the shortest path found with djikstra
    frame : np.ndarray, optional
        frame to show
    ax : plt.axes.Axes, optional
    """
    if ax is None:
        w, h = 1280, 720
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = ['r', 'w', 'g', 'y']

    trajectories_info = fitting_info['trajectories']

    seed_sequence = [t['k_seed'] for t in trajectories_info]
    for i, node in enumerate(sequence):
        k_max = None
        i_max = None
        if link_trajectories and i < len(sequence)-1:
            next_node = sequence[i+1]

            k_min = trajectories_info[seed_sequence.index([node])]['k_min']
            k_max = trajectories_info[seed_sequence.index([node])]['k_max']
            k_min_next = trajectories_info[seed_sequence.index([next_node])]['k_min']

            if k_min >= k_min_next:
                continue

            k_max = k_min_next
            i_max = trajectories_info[seed_sequence.index([next_node])]['i_min']

        show_single_trajectory(fitting_info, candidates, node, k_max=k_max, i_max=i_max, ax=ax, trajectory_color=colors[i%len(colors)], **kwargs)

    if frame is not None:
        ax.imshow(frame, zorder=-100)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        if heatmap is not None:
            ax.imshow(cv2.resize(heatmap, (frame.shape[1], frame.shape[0])), cmap='gray', vmin=0, vmax=1, zorder=-99, alpha=0.7, dpi=dpi)

    return ax


def show_neighboring_trajectories(frame_idx,
                                  fitting_info,
                                  candidates,
                                  path_mapping,
                                  frame,
                                  heatmap=None,
                                  num_prev=2,
                                  num_next=3,
                                  display='params k_min k_max',
                                  display_prev=None,
                                  display_next=None,
                                  color='#FFFFFF',
                                  color_prev='#FF914D',
                                  color_next='#83A83B',
                                  alpha=1,
                                  alpha_prev=0.6,
                                  alpha_next=0.6,
                                  ax=None,
                                  **kwargs):
    # show heatmap and detection candidates
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        if len(heatmap.shape)==2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # put red dot on the frame
    starting_frame = fitting_info['trajectories'][0]['k_seed']

    positions = candidates[frame_idx - starting_frame][:,::-1]
    # positions = positions[np.where(positions[0]>=0)]
    frame = annotate_detections(frame, positions)

    node = path_mapping[frame_idx]

    prev_nodes = find_prev_nodes(frame_idx, path_mapping, num_prev)
    next_nodes = find_next_nodes(frame_idx, path_mapping, num_next)

    for prev_node in prev_nodes:
        ax = show_single_trajectory(fitting_info,
                                    candidates,
                                    prev_node,
                                    display=display_prev,
                                    alpha=alpha_prev,
                                    trajectory_color=color_prev,
                                    ax=ax,
                                    verbose=False,
                                    **kwargs)
    for next_node in next_nodes:
        ax = show_single_trajectory(fitting_info,
                                    candidates,
                                    next_node,
                                    display=display_next,
                                    alpha=alpha_next,
                                    trajectory_color=color_next,
                                    ax=ax,
                                    verbose=False,
                                    **kwargs)

    ax = show_single_trajectory(fitting_info,
                                candidates,
                                node,
                                display=display,
                                frame=frame,
                                frame_index=frame_idx,
                                alpha=alpha,
                                trajectory_color=color,
                                ax=ax,
                                verbose=False,
                                **kwargs)

    im = figure_to_array(ax.figure)

    im2 = np.zeros(im.shape, dtype=im.dtype)
    s = -1
    im2[:-s] = im[s:]
    im2[-s:] = im[:s]
    im2[:1,] = 0

    return im2


def create_trajectory_video(candidates: np.ndarray,
                            src: Union[str, list],
                            dst: str,
                            fitting_info: dict = None,
                            path_mapping: dict = None,
                            heatmaps_folder: str = None,
                            trigger_frames: list = None,
                            num_frames: int = None,
                            starting_frame: int = 0,
                            line_style='-',
                            fitting_kw: dict = {},
                            dpi: int = 100,
                            output_resolution: tuple[int] = (1280, 720),
                            fps: int = 30,
                            **kwargs):
    """Create trajectory video. If num_frames is 0 or 1, an image will be created.

    Parameters
    ----------
    candidates : np.ndarray,
        positions of the detection candidates, of
        shape (num_candidates, max_candidates_per_frame, 2).
        The y and x components are the first and second elements
        of the last dimension, respectively.
    src : Union[str, list]
        source of the video. Can be a collection of images or a video file.
        If a list is passed, multiple sources will be concatenated.
    dst : str
        destination of the video. If `num_frames` is 0 or 1, an image will be created.
    fitting_info : dict, optional
        fitting information provided by `trajectories.fitting.fit_trajectories`.
        If not provided, it will be computed from `candidates` and `n_candidates`.
    path_mapping : dict, optional
        _description_, by default None
    heatmaps_folder : bool, optional
        _description_, by default True
    trigger_frames : list, optional
        _description_, by default None
    num_frames : int, optional
        _description_, by default None
    starting_frame : int, optional
        _description_, by default 0
    line_style : str, optional
        _description_, by default '-'
    fitting_kw : dict, optional
        _description_, by default {}
    dpi : int, optional
        _description_, by default 100
    output_resolution : tuple[int], optional
        _description_, by default (1280, 720)
    fps : int, optional
        _description_, by default 30

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    if fitting_info is None:
        fitting_info = fit_trajectories(candidates, starting_frame, **fitting_kw)
        trajectory_graph = build_trajectory_graph(fitting_info)
        shortest_paths = find_shortest_paths(trajectory_graph)
    if path_mapping is None:
        path_mapping = build_path_mapping(fitting_info, shortest_paths)

    print("Rendering video")

    # set dst resolution and fps
    if type(src) is str:
        src = [src]
    if os.path.isfile(src[0]):
        cap = cv2.VideoCapture(src[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        _, first_frame = cap.read()
        cap.release()
    elif os.path.isdir(src[0]):
        first_frame = next(iter(frame_generator(src[0], verbose=False)))
    else:
        raise FileNotFoundError(f'The specified source {src} does not exist')

    if first_frame is not None:
        h, w = first_frame.shape[0], first_frame.shape[1]
    else:
        w, h = output_resolution

    if num_frames is None:
        num_frames = len(candidates)

    output_video = num_frames > 1
    if output_video:
        out = cv2.VideoWriter(filename=dst,
                              fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                              fps=fps,
                              frameSize=(w, h))
        num_frames = min(num_frames, list(path_mapping.keys())[-1] - starting_frame)
    else:
        num_frames = 1

    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    # clear_frame = np.zeros(first_frame.shape, dtype=np.uint8)
    for i, frame in enumerate(frame_generator(src, starting_frame, starting_frame+num_frames)):
        frame = cv2.resize(frame.copy(), (w, h))
        # frame = clear_frame
        ax.cla()
        if i%100 == 0:
            gc.collect()

        heatmap=None
        if trigger_frames is not None:
            heatmap = get_heatmap(trigger_frames, i+starting_frame, w, h)
        im2 = show_neighboring_trajectories(i+starting_frame,
                                            fitting_info,
                                            candidates,
                                            path_mapping,
                                            frame,
                                            heatmap,
                                            ax=ax,
                                            line_style=line_style,
                                            linewidth=1.5,
                                            **kwargs)
        if heatmaps_folder is not None:
            heatmap = get_heatmap_from_folder(heatmaps_folder, i+starting_frame, w, h, zfill=4)
            im2 = cv2.add(im2, heatmap)

        if output_video:
            out.write(cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
    if output_video:
        out.release()
        plt.close(fig)
    else:
        ax.cla()
        ax.imshow(im2)
        if dst is not None:
            fig.savefig(dst)
        return fig, ax

    print("Done.")
