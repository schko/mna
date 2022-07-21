import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cateyes import (plot_segmentation, plot_trajectory,
                    classify_nslr_hmm, classify_remodnav,
                    discrete_to_continuous, continuous_to_discrete)
from cateyes import sample_data_path



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
    if detect_blink:
        blinks_nan = list(sub_df[sub_df[f"{eye_channel} Openness"] < blink_threshold].index)
        blinks_nan = contiguous_intervals(blinks_nan)

    # interpolate NaN rows
    sub_df = sub_df.fillna(sub_df.mean())

    if 'NSLR' in classifiers:
        segment_id, segment_class = classify_nslr_hmm(sub_df['x_deg'], sub_df['y_deg'], sub_df['timestamp'],
                                                      optimize_noise=True)
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


def plot_segments(eye_data, ppid, session, block, number_in_block, timestamp_start, timestamp_end, classifiers):
    # convert continuous ids and descriptions to discrete timepoints and descriptions
    subset_eye_data = eye_data[eye_data['timestamp'] <= timestamp_end]
    for classifier in classifiers:
        (seg_time, seg_class) = continuous_to_discrete(subset_eye_data['timestamp'],
                                                       subset_eye_data[classifier + "_Segment"],
                                                       subset_eye_data[classifier + "_Class"])
        # plot the classification results
        fig, axes = plt.subplots(2, figsize=(15, 6), sharex=True)
        fig.suptitle(classifier, fontsize=10)
        plot_segmentation(eye_data['x_deg'], eye_data['timestamp'], segments=(seg_time, seg_class), events=None,
                          ax=axes[0], color_dict=color_dict)
        plot_segmentation(eye_data['y_deg'], eye_data['timestamp'], segments=(seg_time, seg_class), events=None,
                          ax=axes[1],
                          show_legend=False, color_dict=color_dict);
        plt.xlabel('time (sec)', fontsize=18)
        plt.savefig(
            f"results/{ppid}_{session}_{block}_{number_in_block}_{classifier}_segments_{timestamp_start}s_{timestamp_end}.png")
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
            f"results/{ppid}_{session}_{block}_{number_in_block}_saccade_masks_{timestamp_start}s_{timestamp_end}.png")
        plt.close()

        df_melt = subset_eye_data[["NSLR_Class", "REMODNAV_Class"]].melt(var_name="Classifier",
                                                                         value_name="Prediction")
        sns.countplot(x="Prediction", hue="Classifier", data=df_melt);
        plt.savefig(
            f"results/ppid_{ppid}_session_{session}_block_{block}_trial_{number_in_block}_classifier_counts_{timestamp_start}s_{timestamp_end}.png")
        plt.close()
