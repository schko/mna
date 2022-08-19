from mna.modalities.eye_process import plot_segments, continuous_to_discrete, process_eye, clean_and_classify, coords_to_degree
import pandas as pd
from collections import defaultdict
import numpy as np

def process_session_eye(rns_data, event_df, eye_channel='Unity_ViveSREyeTracking', detect_blink=True, plot_frequency=20,
                        plot_eye_snippet=40, save_path='../output/', classifiers=None, plot_eye_result=False):
    """
    event_df: dataframe with timestamps start and end to iterate over
    ecg_channel:
    plot_frequency: plot how many other trials?
    plot_ecg_snippet: plot how many seconds per plot?
    """

    if classifiers is None:
        classifiers = ['NSLR', 'REMODNAV']

    df = pd.DataFrame(rns_data[eye_channel][0], columns=rns_data[eye_channel][1],
                      index=rns_data[eye_channel][2]['ChannelNames']).T
    eye_start_time = df.index[0]
    eye_end_time = df.index[-1]
    count = 0
    eye_results = defaultdict(list)
    class_onsets = defaultdict(list)
    for index, row in event_df.iterrows():
        if plot_frequency > 0 and count % plot_frequency == 0:
            plot_eye_result = True
        else:
            plot_eye_result = False
        timestamp_start = row['trial_start_time']
        timestamp_end = row['trial_end_time']

        timestamp_start = max(timestamp_start, eye_start_time)  # in case we don't have data starting from e.g. 0
        timestamp_end = min(timestamp_end, eye_end_time)  # in case the trial marker ends before eye data

        if plot_eye_result:
            plot_timestamp_end = min(timestamp_start + plot_eye_snippet, timestamp_end)  # plot 30 seconds of data
        else:
            plot_timestamp_end = timestamp_end

        eye_data, intervals_nan = process_eye(df, detect_blink=detect_blink, timestamp_start=timestamp_start,
                                              timestamp_end=timestamp_end, eye_channel='L',
                                              classifiers=classifiers)

        if eye_data.shape[0] == 0 or (
                len(intervals_nan) == 1 and intervals_nan[0][1] == eye_data.timestamp.iloc[-1] and intervals_nan[0][
            0] == eye_data.timestamp.iloc[0]):
            for classifier in classifiers:
                eye_results[classifier].append({})
                class_onsets[classifier].append({})
            continue

        if plot_eye_result:
            plot_segments(eye_data, row.ppid, row.session, row.block, row.number_in_block, timestamp_start,
                          timestamp_end=plot_timestamp_end, classifiers=classifiers, save_path=save_path)
        for classifier in classifiers:
            (seg_time, seg_class) = continuous_to_discrete(eye_data['timestamp'],
                                                           eye_data[classifier + "_Segment"],
                                                           eye_data[classifier + "_Class"])
            seg_df = pd.DataFrame([seg_time, seg_class]).T
            seg_df.columns = ['seg_time', 'seg_class']
            class_onsets[classifier].append(seg_df.values.tolist())
            seg_df.seg_time = pd.to_numeric(seg_df.seg_time)
            seg_df.loc[:, 'duration'] = seg_df['seg_time'].diff().shift(-1)
            seg_df['duration'].iloc[-1] = eye_data['timestamp'].iloc[-1] - seg_df['seg_time'].iloc[-1]
            seg_df_summary = seg_df[['seg_time', 'seg_class', 'duration']].groupby('seg_class').describe()
            seg_df_summary = pd.DataFrame(
                seg_df_summary[[('seg_time', 'count'), ('seg_time', 'min'), ('duration', 'mean')]])
            seg_df_summary.columns = seg_df_summary.columns.droplevel(0)
            seg_df_summary.columns = [f"{classifier}_count", f"{classifier}_first_onset", f"{classifier}_mean_duration"]
            eye_results[classifier].append(seg_df_summary.to_dict())
        count += 1
    post_processed_event_df = event_df.copy()
    for classifier in classifiers:
        eye_results_df = pd.json_normalize(eye_results[classifier])
        eye_results_df[f'{classifier}_class_onsets'] = class_onsets[classifier]
        post_processed_event_df = pd.concat([post_processed_event_df, eye_results_df], axis=1)

    # Pupil diameter per trial
    L_Pupil_Diameter_trial = []
    R_Pupil_Diameter_trial = []

    for index in event_df.index:
        L_Pupil_Diameter_avg = (df['L Pupil Diameter'][(df.index >= event_df['trial_start_time'][index]) &
                                             (df.index <= event_df['trial_end_time'][index]) &
                                                       (df['L Pupil Diameter'] > -1)]).mean()
        R_Pupil_Diameter_avg = (df['R Pupil Diameter'][(df.index >= event_df['trial_start_time'][index]) &
                                             (df.index <= event_df['trial_end_time'][index]) &
                                                       (df['R Pupil Diameter'] > -1)]).replace(-1, np.nan).mean()
        L_Pupil_Diameter_trial.append(L_Pupil_Diameter_avg)
        R_Pupil_Diameter_trial.append(R_Pupil_Diameter_avg)

    L_Pupil_Diameter_trial_avg = pd.DataFrame(L_Pupil_Diameter_trial, columns=["Left Pupil Diameter"])
    R_Pupil_Diameter_trial_avg = pd.DataFrame(R_Pupil_Diameter_trial, columns=["Right Pupil Diameter"])

    post_processed_event_df = pd.concat([post_processed_event_df,
                                         L_Pupil_Diameter_trial_avg, R_Pupil_Diameter_trial_avg], axis = 1)

    return post_processed_event_df

def process_eye_trial_xlsx(xlsx_filename, save_path='../output/', classifiers='NSLR', viewing_dist=65,
                           screen_max_x=1280, screen_max_y=960, plot_eye_result = False, time_units='s',
                           pupil_channel='pupil'):
    """
    A session in this case is actually a full trial.
    """
    if isinstance(classifiers, str):
        classifiers = [classifiers]
    fig = None
    wilming_data = pd.read_excel(xlsx_filename)

    if (wilming_data[pupil_channel] == 0).all(axis=0):  # we have no usable data
        raise Exception("No usable data found.")

    if time_units == 'ms':
        wilming_data.time = wilming_data.time/1000

    wilming_data['x_deg'] = coords_to_degree(wilming_data['x'], viewing_dist=viewing_dist, screen_max=screen_max_x,
                                             screen_min=None).T
    wilming_data['y_deg'] = coords_to_degree(wilming_data['y'], viewing_dist=viewing_dist, screen_max=screen_max_y,
                                             screen_min=None).T
    wilming_data = wilming_data.rename(columns={'time': 'timestamp'})
    eye_data, intervals_nan = clean_and_classify(wilming_data, classifiers=classifiers, detect_blink=True,
                                                 blink_threshold=0, eye_channel='pupil')
    eye_results = defaultdict(list)
    class_onsets = defaultdict(list)
    if eye_data.shape[0] == 0 or (
            len(intervals_nan) == 1 and intervals_nan[0][1] == eye_data.timestamp.iloc[-1] and intervals_nan[0][
        0] == eye_data.timestamp.iloc[0]):
        for classifier in classifiers:
            eye_results[classifier].append({})
            class_onsets[classifier].append({})

    if plot_eye_result:
        fig = plot_segments(eye_data, classifiers=classifiers, save_path=save_path)

    for classifier in classifiers:
        (seg_time, seg_class) = continuous_to_discrete(eye_data['timestamp'],
                                                       eye_data[classifier + "_Segment"],
                                                       eye_data[classifier + "_Class"])
        seg_df = pd.DataFrame([seg_time, seg_class]).T
        seg_df.columns = ['seg_time', 'seg_class']
        class_onsets[classifier].append(seg_df.values.tolist())
        seg_df.seg_time = pd.to_numeric(seg_df.seg_time)
        seg_df.loc[:, 'duration'] = seg_df['seg_time'].diff().shift(-1)
        seg_df['duration'].iloc[-1] = eye_data['timestamp'].iloc[-1] - seg_df['seg_time'].iloc[-1]
        seg_df_summary = seg_df[['seg_time', 'seg_class', 'duration']].groupby('seg_class').describe()
        seg_df_summary = pd.DataFrame(
            seg_df_summary[[('seg_time', 'count'), ('seg_time', 'min'), ('duration', 'mean')]])
        seg_df_summary.columns = seg_df_summary.columns.droplevel(0)
        seg_df_summary.columns = [f"{classifier}_count", f"{classifier}_first_onset", f"{classifier}_mean_duration"]
        eye_results[classifier].append(seg_df_summary.to_dict())
    post_processed_event_df = wilming_data.copy()
    for classifier in classifiers:
        eye_results_df = pd.json_normalize(eye_results[classifier])
        eye_results_df[f'{classifier}_class_onsets'] = class_onsets[classifier]
        post_processed_event_df = pd.concat([post_processed_event_df, eye_results_df], axis=1)
    return eye_data, intervals_nan, eye_results_df.T, fig