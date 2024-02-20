import io
import os
import cv2

import numpy as np
from typing import Union


def figure_to_array(fig):
    """Convert matplotlib figure to numpy array"""
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im[:,:,:3]


def frame_generator_dir(dir_name, start_frame=0, stop_frame=None, verbose=True):
    """Generator that yields frames from a directory containing images.

    Parameters
    ----------
    filename : string
        name of the video file.
    start_frame : int, optional
        starting frame from which to read the video, by default 0
    stop_frame : int, optional
        final frame from which to read the video, by default the final frame
    verbose : bool, optional
        by default True

    Yields
    ------
    array
        the current video frame. The channel order is RGB.

    Raises
    ------
    FileNotFoundError
        if the video file does not exist
    ValueError
        if start_frame >= stop_frame
    """
    if not os.path.exists(dir_name):
        raise FileNotFoundError(f'Folder {dir_name} not found!')

    frame_names = sorted(os.listdir(dir_name))

    if stop_frame is None:
        stop_frame = len(frame_names) - start_frame

    if start_frame >= stop_frame:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    for i, frame_name in enumerate(frame_names[start_frame : stop_frame+1]):
        if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80), end = '\r')
        yield cv2.cvtColor(cv2.imread(os.path.join(dir_name, frame_name)), cv2.COLOR_BGR2RGB)

    if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80))
    if verbose: print("Finished frames.")


def frame_generator_file(filename, start_frame=0, stop_frame=None, verbose=True):
    """Generator that yields frames from a video file.

    Parameters
    ----------
    filename : string
        name of the video file.
    start_frame : int, optional
        starting frame from which to read the video, by default 0
    stop_frame : int, optional
        final frame from which to read the video, by default the final frame
    verbose : bool, optional
        by default True

    Yields
    ------
    array
        the current video frame. The channel order is RGB.

    Raises
    ------
    FileNotFoundError
        if the video file does not exist
    ValueError
        if start_frame >= stop_frame
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {filename} not found!')

    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= stop_frame:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    for i in range(start_frame, stop_frame):
        if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80), end = '\r')
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ret, frame = cap.read()
        if not ret:
            if verbose: print("Finished prematurely".ljust(80))
            break
    if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80))
    if verbose: print("Finished frames.")
    cap.release()


def frame_generator(src: Union[str, list],
                    start_frame: int = 0,
                    stop_frame: int = None,
                    verbose: bool = True):
    """Generator that yields video frames from the given sources.

    Parameters
    ----------
    src : string | list of str
        names of the video sources (one or multiple). Each source can either be a video file
        or a folder containing the single video frames as image files.
    start_frame : int, optional
        starting frame from which to read the video sources, by default 0.
    stop_frame : int | list of int, optional
        final frame from which to read the video sources, by default the final frame
    verbose : bool, optional
        by default True

    Yields
    ------
    array
        the current video frame. The channel order is RGB.

    Raises
    ------
    FileNotFoundError
        if a video source does not exist
    ValueError
         - if `sources`, `start_frames` and `stop_frames` have different lengths
         - if `start_frames[i]` >= `stop_frames[i]` for any `i`
    """
    if type(src) is str:
        src = [src]

    for source in src:
        if not os.path.exists(source):
            raise ValueError(f'Source {src} not found')

    frame_counter = 0
    for i in range(len(src)):
        if stop_frame is not None and frame_counter >= stop_frame-start_frame:
            break
        if verbose: print(f'source {i+1} of {len(src)}:')
        generator = frame_generator_dir if os.path.isdir(src[i]) else frame_generator_file

        stop = stop_frame - frame_counter if stop_frame is not None else None
        for frame in generator(src[i], start_frame, stop, verbose):
            yield frame
            frame_counter += 1
        start_frame = 0
