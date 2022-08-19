import os
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from heartpy import exceptions
import heartpy as hp
from heartpy import config
import matplotlib.pyplot as plt
def ecg_plotter(working_data, measures, start_time=0, show=True, figsize=None,
            title='Heart Rate Signal Peak Detection', moving_average=False):  # pragma: no cover
    '''plots the analysis results.
    Function that uses calculated measures and data stored in the working_data{} and measures{}
    dict objects to visualise the fitted peak detection solution.
    Parameters
    ----------
    working_data : dict
        dictionary object that contains all heartpy's working data (temp) objects.
        will be created if not passed to function
    measures : dict
        dictionary object used by heartpy to store computed measures. Will be created
        if not passed to function
    show : bool
        when False, function will return a plot object rather than display the results.
        default : True
    figsize: tuple
        Set dimensions of image in inches like in matplotlib. figsize=(x, y)
        default: None => (6.4, 4.8)
    title : string
        title for the plot.
        default : "Heart Rate Signal Peak Detection"
    moving_average : bool
        whether to display the moving average on the plot.
        The moving average is used for peak fitting.
        default: False
    Returns
    -------
    out : matplotlib plot object
        only returned if show == False.
    Examples
    --------
    First let's load and analyse some data to visualise
    >>> import heartpy as hp
    >>> data, _ = hp.load_exampledata(0)
    >>> wd, m = hp.process(data, 100.0)
    Then we can visualise
    >>> plot_object = plotter(wd, m, show=False, title='some awesome title')
    This returns a plot object which can be visualized or saved or appended.
    See matplotlib API for more information on how to do this.
    A matplotlib plotting object is returned. This can be further processed and saved
    to a file.
    '''
    # get color palette
    colorpalette = config.get_colorpalette_plotter()

    # create plot x-var
    fs = working_data['sample_rate']
    plotx = np.arange(start_time, start_time + len(working_data['hr']) / fs, 1 / fs)
    # check if there's a rounding error causing differing lengths of plotx and signal
    diff = len(plotx) - len(working_data['hr'])
    if diff < 0:
        # add to linspace
        plotx = np.append(plotx, plotx[-1] + (plotx[-2] - plotx[-1]))
    elif diff > 0:
        # trim linspace
        plotx = plotx[0:-diff]

    peaklist = working_data['peaklist']
    ybeat = working_data['ybeat']
    rejectedpeaks = working_data['removed_beats']
    rejectedpeaks_y = working_data['removed_beats_y']
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)
    ax.plot(plotx, working_data['hr'], color=colorpalette[0], label='heart rate signal', zorder=-10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('mV')

    if moving_average:
        ax.plot(plotx, working_data['rolling_mean'], color='gray', alpha=0.5)

    ax.scatter(np.asarray(peaklist) / fs, ybeat, color=colorpalette[1], label='BPM:%.2f' % (measures['bpm']))
    ax.scatter(rejectedpeaks / fs, rejectedpeaks_y, color=colorpalette[2], label='rejected peaks')

    # check if rejected segment detection is on and has rejected segments
    try:
        if len(working_data['rejected_segments']) >= 1:
            for segment in working_data['rejected_segments']:
                ax.axvspan(segment[0], segment[1], facecolor='red', alpha=0.5)
    except:
        pass

    ax.legend(loc=4, framealpha=0.6)

    if show:
        fig.show()
    else:
        return fig


def process_ecg(df, freq, low_bpm=40, high_bpm=200, timestamp_start=0, timestamp_end=None, ecg_channel='EX2',
                flip_if_failed=True):
    """
    low_bpm: minimum bpm possible before classifying as a failed detection
    max_bpm: maximum bpm possible before classifying as a failed detection
    timestamp_start: inclusive seconds
    timestamp_end: inclusive seconds
    """
    if not timestamp_end:
        timestamp_end = df.index[-1]
    ecg_data = df['EX2'][(df.index >= timestamp_start) & (df.index <= timestamp_end)].values
    flip_if_failed = flip_if_failed
    flipped_signal = False
    failed_to_detect = False
    working_data, measures = None, None
    try:
        working_data, measures = hp.process(ecg_data, freq)
    except exceptions.BadSignalWarning as e:
        failed_to_detect = True
        pass
    except IndexError as e:
        failed_to_detect = True
        return working_data, measures, flipped_signal, failed_to_detect
    if not working_data:
        failed_to_detect = True
    # check if BPM is within range using valid beats
    if not failed_to_detect:
        valid_beats_bpm = ((len(working_data['peaklist']) - len(working_data['removed_beats'])) / (
                    timestamp_end - timestamp_start)) * 60
        if not failed_to_detect and ((valid_beats_bpm < low_bpm) or (valid_beats_bpm > high_bpm)):
            failed_to_detect = True

    if failed_to_detect and flip_if_failed:  # try flipping the signal
        try:
            flipped_ecg = hp.flip_signal(ecg_data)
            flipped_signal = True
            working_data, measures = hp.process(flipped_ecg, freq)
            failed_to_detect = False
        except exceptions.BadSignalWarning as e:
            pass
    if not working_data: # we haven't found anything
        return working_data, measures, flipped_signal, failed_to_detect
    # check if BPM is within range using valid beats
    if not failed_to_detect:
        valid_beats_bpm = ((len(working_data['peaklist']) - len(working_data['removed_beats'])) / (
                    timestamp_end - timestamp_start)) * 60
        if ((valid_beats_bpm < low_bpm) or (valid_beats_bpm > high_bpm)):
            failed_to_detect = True

    # add back in the offset, note these beats are in samples
    working_data['peaklist'] = np.array(working_data['peaklist']) + (timestamp_start * freq)
    working_data['removed_beats'] = working_data['removed_beats'] + timestamp_start * freq
    return working_data, measures, flipped_signal, failed_to_detect


def plot_ecg(df, working_data, measures, timestamp_start=0, timestamp_end=None):
    """
    timestamp_start: inclusive seconds
    timestamp_end: inclusive seconds
    """

    def snip_arr(arr, end_time, freq):
        time_converted = sum((np.asarray(arr) / freq) <= end_time)  # this will be sorted so quick get of bound
        return arr[:time_converted]

    if timestamp_end:
        working_data_snippet = {}
        df = df[df.index <= timestamp_end]
        end_time = df.index[-1]
        working_data_snippet['sample_rate'] = working_data['sample_rate']
        working_data_snippet['hr'] = working_data['hr'][:len(df)]
        working_data_snippet['peaklist'] = snip_arr(working_data['peaklist'], end_time, working_data['sample_rate'])
        working_data_snippet['ybeat'] = working_data['ybeat'][:len(working_data_snippet['peaklist'])]
        working_data_snippet['removed_beats'] = snip_arr(working_data['removed_beats'], end_time,
                                                         working_data['sample_rate'])
        working_data_snippet['removed_beats_y'] = working_data['removed_beats_y'][
                                                  :len(working_data_snippet['removed_beats'])]
    plot_object = ecg_plotter(working_data_snippet, measures, start_time=timestamp_start, show=False, title='ECG')
    return plot_object