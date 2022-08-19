import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cateyes import (plot_segmentation, continuous_to_discrete)
from cateyes import coords_to_degree as ctod

# Calculate vergence, combined eye gaze data

def calc_vergence_from_dir(L0, R0, LDir, RDir):
    # Calculating shortest line segment intersecting both lines
    # Implementation sourced from http://paulbourke.net/geometry/pointlineplane/
    L0R0 = L0 - R0  # segment between L origin and R origin
    epsilon = 0.00000001  # small positive real number

    # Calculating dot-product equation to find perpendicular shortest-line-segment
    d1343 = L0R0[0] * RDir[0] + L0R0[1] * RDir[1] + L0R0[2] * RDir[2]
    d4321 = RDir[0] * LDir[0] + RDir[1] * LDir[1] + RDir[2] * LDir[2]
    d1321 = L0R0[0] * LDir[0] + L0R0[1] * LDir[1] + L0R0[2] * LDir[2]
    d4343 = RDir[0] * RDir[0] + RDir[1] * RDir[1] + RDir[2] * RDir[2]
    d2121 = LDir[0] * LDir[0] + LDir[1] * LDir[1] + LDir[2] * LDir[2]
    denom = d2121 * d4343 - d4321 * d4321
    if abs(denom) < epsilon:
        return 1.0  # no intersection, would cause div by 0 err (potentially)
    numer = d1343 * d4321 - d1321 * d4343

    # calculate scalars (mu) that scale the unit direction XDir to reach the desired points
    # variable scale of direction vector for LEFT ray
    muL = numer / denom
    # variable scale of direction vector for RIGHT ray
    muR = (d1343 + d4321 * (muL)) / d4343

    # calculate the points on the respective rays that create the intersecting line
    ptL = L0 + muL * LDir  # the point on the Left ray
    ptR = R0 + muR * RDir  # the point on the Right ray

    # calculate the vector between the middle of the two endpoints and return its magnitude
    # middle point between two endpoints of shortest-line-segment
    ptM = (ptL + ptR) / 2.0
    oM = (L0 + R0) / 2.0  # midpoint between two (L & R) origins
    FinalRay = ptM - oM  # Combined ray between midpoints of endpoints
    # returns the magnitude of the vector (length)
    return oM[0], oM[1], oM[2], FinalRay[0], FinalRay[1], FinalRay[2], np.linalg.norm(FinalRay) / 100.0


def combine_gaze(sub_df):
    sub_df[['Combined Gaze Origin X', 'Combined Gaze Origin Y', 'Combined Gaze Origin Z', 'Combined Gaze Direction X',
            'Combined Gaze Direction Y', 'Combined Gaze Direction Z', 'Combined Gaze Magnitude']] = sub_df.apply(
        lambda row: calc_vergence_from_dir(
            np.column_stack((row['L Gaze Origin X'], row['L Gaze Origin Y'], row['L Gaze Origin Z']))[0],
            np.column_stack((row['R Gaze Origin X'], row['R Gaze Origin Y'], row['R Gaze Origin Z']))[0],
            np.column_stack((row['L Gaze Direction X'], row['L Gaze Direction Y'], row['L Gaze Direction Z']))[0],
            np.column_stack((row['R Gaze Direction X'], row['R Gaze Direction Y'], row['R Gaze Direction Z']))[0]),
        axis=1, result_type='expand')
    return sub_df


def get_x_y_deg(sub_df, eye='L'):
    sub_df['x_deg'] = np.arctan(sub_df[eye + ' Gaze Direction X'] / sub_df[eye + ' Gaze Direction Z']) / np.pi * 180
    sub_df['y_deg'] = np.arctan(sub_df[eye + ' Gaze Direction Y'] / sub_df[eye + ' Gaze Direction Z']) / np.pi * 180
    return sub_df


color_dict = {'Fixation': 'blue',
              'Saccade': 'black',
              'ISaccade': 'gray',
              'Smooth Pursuit': 'green',
              'PSO': 'yellow',
              'High-Velocity PSO': 'gold',
              'Low-Velocity PSO': 'greenyellow',
              'High-Velocity PSO (NCB)': 'goldenrod',
              'Low-Velocity PSO (NCB)': 'yellowgreen',
              'None': 'grey',
              None: 'grey',
              'Blink': 'red'}


def clean_and_classify(sub_df, classifiers, detect_blink=True, blink_threshold=.40, eye_channel='L'):
    def contiguous_intervals(nums):
        nums.sort()
        nums.append(1e9)
        ans = []
        l = nums[0]
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:
                ans.append([l, nums[i - 1]])
                l = nums[i]
        return ans

    indices_nan = list(sub_df.reset_index()[sub_df.isnull().any(axis=1)].index)
    intervals_nan = contiguous_intervals(list(indices_nan))
    blinks_nan = []
    if detect_blink and f"{eye_channel} Openness" in sub_df:
        blinks_nan = list(sub_df[sub_df[f"{eye_channel} Openness"] < blink_threshold].index)
        blinks_nan = contiguous_intervals(blinks_nan)

    # interpolate NaN rows
    sub_df = sub_df.fillna(sub_df.mean())

    if 'NSLR' in classifiers:
        segment_id, segment_class = classify_nslr_hmm(sub_df['x_deg'], sub_df['y_deg'],
                                                      sub_df['timestamp'], optimize_noise=True)
        segment_class = np.array(segment_class)
        for interval in intervals_nan:
            segment_class[interval[0]:interval[1]] = None
            segment_id[interval[1]:] = segment_id[interval[1]:] + 1
        if detect_blink:
            for interval in blinks_nan:
                segment_class[interval[0]:interval[1]] = 'Blink'
                segment_id[interval[1]:] = segment_id[interval[1]:] + 1
        sub_df["NSLR_Segment"] = segment_id
        sub_df["NSLR_Class"] = segment_class

    if 'REMODNAV' in classifiers:
        try:
            segment_id, segment_class = classify_remodnav(sub_df['x_deg'], sub_df['y_deg'], sub_df['timestamp'],
                                                          px2deg=1.,
                                                          preproc_kwargs={'savgol_length': .03}, simple_output=True)
            segment_class = np.array(segment_class)
        except ValueError as e:
            segments = np.full([len(sub_df['timestamp'])], 0)
            segment_class = np.full([len(sub_df['timestamp'])], None)

        for interval in intervals_nan:
            segment_class[interval[0]:interval[1]] = None
            segment_id[interval[1]:] = segment_id[interval[1]:] + 1
        if detect_blink:
            for interval in blinks_nan:
                segment_class[interval[0]:interval[1]] = 'Blink'
                segment_id[interval[1]:] = segment_id[interval[1]:] + 1
        sub_df["REMODNAV_Segment"] = segment_id
        sub_df["REMODNAV_Class"] = segment_class

    # convert intervals to timestamps
    intervals_nan = [[sub_df.iloc[i[0]].timestamp, sub_df.iloc[i[1]].timestamp] for i in intervals_nan]
    return sub_df, intervals_nan


def process_eye(df, timestamp_start=0, timestamp_end=None, detect_blink=True, eye_channel='L', classifiers=['NSLR']):
    if not timestamp_end:
        timestamp_end = df.index[-1]
    eye_data = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]

    if eye_channel == 'C':  # if combined, try to calculated combined ray
        try:
            eye_data = combine_gaze(eye_data)
        except ValueError as e:
            eye_data = eye_data.reset_index().rename(columns={'index': 'timestamp'})
            return eye_data, [[eye_data.timestamp.iloc[0], eye_data.timestamp.iloc[-1]]]
    elif (eye_data[f"{eye_channel} Pupil Diameter"] == -1.0).all(axis=0):  # we have no usable data
        eye_data = eye_data.reset_index().rename(columns={'index': 'timestamp'})
        if eye_data.shape[0] == 0:
            return eye_data, [[]]
        return eye_data, [[eye_data.timestamp.iloc[0], eye_data.timestamp.iloc[-1]]]

    eye_data = get_x_y_deg(eye_data, eye=eye_channel)
    eye_data = eye_data.reset_index().rename(columns={'index': 'timestamp'})
    eye_data, intervals_nan = clean_and_classify(eye_data, classifiers, detect_blink=detect_blink,
                                                 eye_channel=eye_channel)
    return eye_data, intervals_nan


def plot_segments(eye_data, ppid='', session=1, block=1, number_in_block=1, timestamp_start=None, timestamp_end=None,
                  classifiers = None, save_path='../output/'):
    if classifiers is None:
        classifiers = ['NSLR', 'REMODNAV']
    fig = None
    # convert continuous ids and descriptions to discrete timepoints and descriptions
    if not timestamp_start:
        timestamp_start = eye_data.timestamp.iat[0]
    if timestamp_end:
        subset_eye_data = eye_data[eye_data['timestamp'] <= timestamp_end]
    else:
        timestamp_end = eye_data.timestamp.iat[-1]
        subset_eye_data = eye_data
    for classifier in classifiers:
        (seg_time, seg_class) = continuous_to_discrete(subset_eye_data['timestamp'],
                                                       subset_eye_data[classifier + "_Segment"],
                                                       subset_eye_data[classifier + "_Class"])
        # plot the classification results
        if not os.path.isdir(save_path): os.makedirs(save_path)
        fig, axes = plt.subplots(2, figsize=(15, 6), sharex=True)
        fig.suptitle(classifier, fontsize=10)
        plot_segmentation(eye_data['x_deg'], eye_data['timestamp'], segments=(seg_time, seg_class), events=None,
                          ax=axes[0], color_dict=color_dict)
        plot_segmentation(eye_data['y_deg'], eye_data['timestamp'], segments=(seg_time, seg_class), events=None,
                          ax=axes[1],
                          show_legend=False, color_dict=color_dict);
        plt.xlabel('time (sec)', fontsize=18)
        plt.savefig(
            f"{save_path}{ppid}_{session}_{block}_{number_in_block}_{classifier}_segments_{timestamp_start}s_{timestamp_end}.png")
        plt.close()

    if 'NSLR' in classifiers and 'REMODNAV' in classifiers:
        # add a saccade mask and plot the data
        subset_eye_data["NSLR_SP"] = (subset_eye_data["NSLR_Class"] == "Saccade") * 1
        subset_eye_data["REMODNAV_SP"] = (subset_eye_data["REMODNAV_Class"] == "Saccade") * 1
        axes = subset_eye_data[["x_deg", "y_deg", "NSLR_SP", "REMODNAV_SP"]].plot(subplots=True, legend=False,
                                                                                  title="Saccade Masks (NSLR and REMODNAV)")

        # set labels, axes, etc...
        axes[0].set_ylabel("Theta")
        axes[1].set_ylabel("Phi")
        axes[2].set_ylabel("NSLR")
        axes[3].set_ylabel("REMODNAV")
        axes[2].yaxis.tick_right()
        axes[2].set_yticklabels(["", "No Saccade", "Saccade"])
        axes[3].yaxis.tick_right()
        axes[3].set_yticklabels(["", "No Saccade", "Saccade"])
        plt.xlabel("Time");
        plt.savefig(
            f"{save_path}{ppid}_{session}_{block}_{number_in_block}_saccade_masks_{timestamp_start}s_{timestamp_end}.png")
        plt.close()

        df_melt = subset_eye_data[["NSLR_Class", "REMODNAV_Class"]].melt(var_name="Classifier",
                                                                         value_name="Prediction")
        sns.countplot(x="Prediction", hue="Classifier", data=df_melt)
        plt.savefig(
            f"{save_path}/ppid_{ppid}_session_{session}_block_{block}_trial_{number_in_block}_classifier_counts_{timestamp_start}s_{timestamp_end}.png")
        plt.close()
    return fig

# NOTE: classification functions had issues with python 3.10 so moved out, specifically, discrete_to_continuous

import numpy as np
import nslr_hmm
from remodnav.clf import EyegazeClassifier
from cateyes.utils import continuous_to_discrete

import warnings

WARN_SFREQ = "\n\nIrregular sampling rate detected. This can lead to impaired " \
             "performance with this classifier. Consider resampling your data to " \
             "a fixed sampling rate. Setting sampling rate to average sample difference."

CLASSES = {nslr_hmm.FIXATION: 'Fixation',
           nslr_hmm.SACCADE: 'Saccade',
           nslr_hmm.SMOOTH_PURSUIT: 'Smooth Pursuit',
           nslr_hmm.PSO: 'PSO',
           None: "None", }

REMODNAV_CLASSES = {"FIXA": "Fixation", "SACC": "Saccade",
                    "ISAC": "Saccade (ISI)", "PURS": "Smooth Pursuit",
                    "HPSO": "High-Velocity PSO",
                    "LPSO": "Low-Velocity PSO",
                    "IHPS": "High-Velocity PSO (ISI)",
                    "ILPS": "Low-Velocity PSO (ISI)"}

REMODNAV_SIMPLE = {"FIXA": "Fixation", "SACC": "Saccade",
                   "ISAC": "Saccade", "PURS": "Smooth Pursuit",
                   "HPSO": "PSO", "LPSO": "PSO",
                   "IHPS": "PSO", "ILPS": "PSO"}

def coords_to_degree(x, viewing_dist, screen_max, screen_min=None):
    """
    Pass-through
    """
    return ctod(x, viewing_dist, screen_max, screen_min=None)

def classify_nslr_hmm(x, y, time, return_discrete=False, return_orig_output=False, **nslr_kwargs):
    """Robust gaze classification using NSLR-HMM by Pekannen & Lappi (2017).

    NSLR-HMM takes eye tracking data (in degree units), segments them using
    Naive Segmented Linear Regression and then categorizes these segments based
    on a pretrained Hidden Markov Model. NSLR-HMM can separate between the
    following classes:
    ```
    Fixation, Saccade, Smooth Pursuit, PSO
    ```

    For more information and documentation, see the [pupil-labs implementation].
    [pupil-labs implementation]: https://github.com/pupil-labs/nslr-hmm

    For reference see:

    ---
    Pekkanen, J., & Lappi, O. (2017). A new and general approach to
    signal denoising and eye movement classification based on segmented
    linear regression. Scientific reports, 7(1), 1-13.
    ---


    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data. Must be
        represented in degree units.
    y : array of float
        A 1D-array representing the y-axis of your gaze data. Must be
        represented in degree units.
    times : array of float
        A 1D-array representing the sampling times of the continuous
        eyetracking recording (in seconds).
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the `times` array). Default=False.
    return_orig_output : bool
        If True, additionally return NSLR-HMM's original segmentation
        object as output. Default=False.
    **nslr_kwargs
        Any additional keyword argument will be passed to
        nslr_hmm.classify_gaze().

    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
    segment_dict : dict
        A dictionary containing the original output from NSLR-HMM:
        "sample_class", "segmentation" and "seg_class". Only returned if
        `return_orig_output = True`.

    """

    # extract gaze and time array
    gaze_array = np.vstack([x, y]).T
    time_array = np.array(time) if hasattr(time, '__iter__') else np.arange(0, len(x), 1 / time)

    # classify using NSLR-HMM
    sample_class, seg, seg_class = nslr_hmm.classify_gaze(time_array, gaze_array,
                                                          **nslr_kwargs)

    # define discrete version of segments/classes
    segments = [s.t[0] for s in seg.segments]
    classes = seg_class

    # convert them if continuous series wanted
    if return_discrete == False:
        segments, classes = discrete_to_continuous(time_array, segments, classes)

    # add the prediction to our dataframe
    classes = [CLASSES[i] for i in classes]

    if return_orig_output:
        # create dictionary from it
        segment_dict = {"sample_class": sample_class, "segmentation": seg, "seg_class": seg_class}
        return segments, classes, segment_dict
    else:
        return segments, classes


def classify_remodnav(x, y, time, px2deg, return_discrete=False, return_orig_output=False,
                      simple_output=False, classifier_kwargs={}, preproc_kwargs={},
                      process_kwargs={}):
    """REMoDNaV robust eye movement prediction by Dar, Wagner, & Hanke (2021).

    REMoDNaV is a fixation-based algorithm which is derived from the Nyström & Holmqvist
    (2010) algorithm, but adds various extension. It aims to provide robust
    classification under different eye tracking settings. REMoDNaV can separate between
    the following classes:
    ```
    Fixation, Saccade, Saccade (intersaccadic interval), Smooth Pursuit,
    High-Velocity PSO, High-Velocity PSO (intersaccadic interval),
    Low-Velocity PSO, Low-Velocity PSO (intersaccadic interval)
    ```
    For information on the difference between normal (chunk boundary) intervals and
    intersaccadic intervals, please refer to the original paper.


    If `simple_output=True`, REMoDNaV will separate between the following classes:
    ```
    Fixation, Saccade, Smooth Pursuit, PSO
    ```

    For more information and documentation, see the [original implementation].
    [original implementation]: https://github.com/psychoinformatics-de/remodnav


    For reference see:

    ---
    Dar, A. H., Wagner, A. S., & Hanke, M. (2021). REMoDNaV: robust eye-movement
    classification for dynamic stimulation. Behavior research methods, 53(1), 399-414.
    DOI: 10.3758/s13428-020-01428-x
    ---
    Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation, saccade,
    and glissade detection in eyetracking data. Behavior research methods,
    42(1), 188-204.
    ---

    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze
        arrays or a float/int that represents the sampling rate.
    px2deg : float
        The ratio between one pixel in the recording and one degree.
        If `x` and `y` are in degree units, px2deg = 1.
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.
    return_orig_output : bool
        If True, additionally return REMoDNaV's original segmentation
        events as output. Default=False.
    simple_output : bool
        If True, return a simplified version of REMoDNaV's output,
        containing only the gaze categories: ["Fixation", "Saccade",
        "Smooth Pursuit", "PSO"]. Default=False.
    classifier_kwargs : dict
        A dict consisting of keys that can be fed as keyword arguments
        to remodnav.clf.EyegazeClassifier(). Default={}.
    preproc_kwargs : dict
        A dict consisting of keys that can be fed as keyword arguments
        to remodnav.clf.EyegazeClassifier().preproc(). Default={}.
    process_kwargs : dict
        A dict consisting of keys that can be fed as keyword arguments
        to remodnav.clf(). Default={}.

    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
    events : array
        A record array containing the original output from REMoDNaV.
        Only returned if `return_orig_output = True`.

    """

    # process time argument
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1])
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time

    # format and preprocess the data
    data = np.core.records.fromarrays([x, y], names=["x", "y"])

    # define the classifier, preprocess data and run the classification
    clf = EyegazeClassifier(px2deg, sfreq, **classifier_kwargs)
    data_preproc = clf.preproc(data, **preproc_kwargs)
    events = clf(data_preproc, **process_kwargs)

    # add the start time offset to the events
    for i in range(len(events)):
        events[i]['start_time'] += times[0]
        events[i]['end_time'] += times[0]

    # extract the classifications
    class_dict = REMODNAV_SIMPLE if simple_output else REMODNAV_CLASSES
    segments, classes = zip(*[(ev["start_time"], class_dict[ev["label"]]) for ev in events])

    # convert them if continuous series wanted
    if return_discrete == False:
        segments, classes = discrete_to_continuous(times, segments, classes)

    # return
    if return_orig_output:
        return segments, classes, events
    else:
        return segments, classes


def classify_velocity(x, y, time, threshold, return_discrete=False):
    """I-VT velocity algorithm from Salvucci & Goldberg (2000).

    One of several algorithms proposed in Salvucci & Goldberg (2000),
    the I-VT algorithm classifies samples as saccades if their rate of
    change from a previous sample exceeds a certain threshold. I-VT
    can separate between the following classes:
    ```
    Fixation, Saccade
    ```

    For reference see:

    ---
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations
    and saccades in eye-tracking protocols. In Proceedings of the
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    ---

    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze
        arrays or a float/int that represents the sampling rate.
    threshold : float
        The maximally allowed velocity after which a sample should be
        classified as "Saccade". Threshold can be interpreted as
        `gaze_units/s`, with `gaze_units` being the spatial unit of
        your eyetracking data (e.g. pixels, cm, degrees).
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.

    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
        """
    # process time argument and calculate sample threshold
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1])
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time
    sample_thresh = threshold / sfreq

    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([[0.], vels])

    # define classes by threshold
    classes = np.empty(len(x), dtype=object)
    classes[:] = "Fixation"
    classes[vels > sample_thresh] = "Saccade"

    # group consecutive classes to one segment
    segments = np.zeros(len(x), dtype=int)
    for idx in range(1, len(classes)):
        if classes[idx] == classes[idx - 1]:
            segments[idx] = segments[idx - 1]
        else:
            segments[idx] = segments[idx - 1] + 1

    # return output
    if return_discrete:
        segments, classes = continuous_to_discrete(times, segments, classes)
    return segments, classes


def classify_dispersion(x, y, time, threshold, window_len, return_discrete=False):
    """I-DT dispersion algorithm from Salvucci & Goldberg (2000).

    The I-DT algorithm classifies fixations by checking if the dispersion of
    samples within a certain window does not surpass a predefined threshold.
    I-DT can separate between the following classes:
    ```
    Fixation, Saccade
    ```

    For reference see:

    ---
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations
    and saccades in eye-tracking protocols. In Proceedings of the
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    ---

    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze
        arrays or a float/int that represents the sampling rate.
    threshold : float
        The maximally allowed dispersion (difference of x/y min and
        max values) within `window_len` in order to be counted as a
        Fixation. Value depends on the unit of your gaze data.
    window_len : float
        The window length in seconds within which the dispersion is
        calculated.
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.

    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
    """

    def _disp(win_x, win_y):
        """Calculate the dispersion of a window."""
        delta_x = np.max(win_x) - np.min(win_x)
        delta_y = np.max(win_y) - np.min(win_y)
        return delta_x + delta_y

    # process time argument
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1])
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time

    # infer number of samples from windowlen
    n_samples = int(sfreq * window_len)

    # per default everything is a saccade
    segments = np.zeros(len(x), dtype=int)
    classes = np.empty(len(x), dtype=object)
    classes[0:] = "Saccade"

    # set start window and segment
    i_start = 0
    i_stop = n_samples
    seg_idx = 0

    while i_stop <= len(x):

        # set the current window
        win_x = x[i_start:i_stop]
        win_y = y[i_start:i_stop]

        # if we're in a Fixation
        if _disp(win_x, win_y) <= threshold:

            # start a fixation segment
            seg_idx += 1

            # as long as we're in the fixation
            while _disp(win_x, win_y) <= threshold and i_stop < len(x):
                # make the chunk larger
                i_stop += 1
                win_x = x[i_start:i_stop]
                win_y = y[i_start:i_stop]

            # mark it
            classes[i_start:i_stop] = "Fixation"
            segments[i_start:i_stop] = seg_idx

            # start looking at a new chunk
            i_start = i_stop
            i_stop = i_stop + n_samples
            seg_idx += 1

        else:
            # move window point further
            segments[i_start:i_stop] = seg_idx
            i_start += 1
            i_stop = i_start + n_samples

    # return output
    if return_discrete:
        segments, classes = continuous_to_discrete(times, segments, classes)
    return segments, classes


def mad_velocity_thresh(x, y, time, th_0=200, return_past_threshs=False):
    """Robust Saccade threshold estimation using median absolute deviation.

    Can be used to estimate a robust velocity threshold to use as threshold
    parameter in the `classify_velocity` algorithm.

    Implementation taken from [this gist] by Ashima Keshava.
    [this gist]: https://gist.github.com/ashimakeshava/ecec1dffd63e49149619d3a8f2c0031f

    For reference, see the paper:

    ---
    Voloh, B., Watson, M. R., König, S., & Womelsdorf, T. (2019). MAD
    saccade: statistically robust saccade threshold estimation via the
    median absolute deviation. Journal of Eye Movement Research, 12(8).
    ---

    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze
        arrays or a float/int that represents the sampling rate.
    th_0 : float
        The initial threshold used at start. Threshold can be interpreted
        as `gaze_units/s`, with `gaze_units` being the spatial unit of
        your eyetracking data (e.g. pixels, cm, degrees). Defaults to 200.
    return_past_thresholds : bool
        Whether to additionally return a list of all thresholds used
        during iteration. Defaults do False.

    Returns
    -------
    threshold : float
        The maximally allowed velocity after which a sample should be
        classified as "Saccade". Threshold can be interpreted as
        `gaze_units/ms`, with `gaze_units` being the spatial unit of
        your eyetracking data (e.g. pixels, cm, degrees).
    past_thresholds : list of float
        A list of all thresholds used during iteration. Only returned
        if `return_past_thresholds` is True.

    Example
    --------
    >>> threshold = mad_velocity_thresh(x, y, time)
    >>> segments, classes = classify_velocity(x, y, time, threshold)
    """
    # process time argument and calculate sample threshold
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1])
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time
    # get init thresh per sample
    th_0 = th_0 / sfreq

    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([[0.], vels])

    # define saccade threshold by MAD
    threshs = []
    angular_vel = vels
    while True:
        threshs.append(th_0)
        angular_vel = angular_vel[angular_vel < th_0]
        median = np.median(angular_vel)
        diff = (angular_vel - median) ** 2
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        th_1 = median + 3 * 1.48 * med_abs_deviation
        # print(th_0, th_1)
        if (th_0 - th_1) > 1:
            th_0 = th_1
        else:
            saccade_thresh = th_1
            threshs.append(saccade_thresh)
            break

    # revert units
    saccade_thresh = saccade_thresh * sfreq
    threshs = [i * sfreq for i in threshs]

    if return_past_threshs:
        return saccade_thresh, threshs
    else:
        return saccade_thresh


def discrete_to_continuous(times, discrete_times, discrete_values):
    """Matches an array of discrete events to a continuous time series.

    Parameters
    ----------
    times : array of (float, int)
        A 1D-array representing the sampling times of the continuous
        eyetracking recording.
    discrete_times : array of (float, int)
        A 1D-array representing discrete timepoints at which a specific
        event occurs. Is used to map `discrete_values` onto `times`.
    discrete_values : array
        A 1D-array containing the event description or values
        corresponding to `discrete_times`. Must be the same length as
        `discrete_times`.

    Returns
    -------
    indices : array of int
        Array of length len(times) corresponding to the event index
        of the discrete events mapped onto the sampling times.
    values : array
        Array of length len(times) corresponding to the event values
        or descriptions of the discrete events.

    Example
    --------
    >>> times = np.array([0., 0.1, 0.2])
    >>> dis_times, dis_values = [0.1], ["Saccade"]
    >>> discrete_to_continuous(times, dis_times, dis_values)
    array([0., 1., 1.]), array([None, 'Saccade', 'Saccade'])
    """

    # sort the discrete events by time
    time_val_sorted = sorted(zip(discrete_times, discrete_values))

    # fill the time series with indices and values
    indices = np.zeros(len(times))
    values = np.empty(len(times), dtype=object)
    for idx, (dis_time, dis_val) in enumerate(time_val_sorted):
        indices[times >= dis_time] = idx + 1
        values[times >= dis_time] = dis_val

    return indices, values