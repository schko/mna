from mna.modalities.eye_process import plot_segments, continuous_to_discrete, process_eye
import pandas as pd


def process_session_eye(rns_data, event_df, eye_channel='Unity_ViveSREyeTracking', detect_blink=True, plot_frequency=20,
                        plot_eye_snippet=40, save_path='../output/'):
    """
    event_df: dataframe with timestamps start and end to iterate over
    ecg_channel:
    plot_frequency: plot how many other trials?
    plot_ecg_snippet: plot how many seconds per plot?
    """

    df = pd.DataFrame(rns_data[eye_channel][0], columns=rns_data[eye_channel][1],
                      index=rns_data[eye_channel][2]['ChannelNames']).T
    classifiers = ['NSLR', 'REMODNAV']
    plot_frequency = plot_frequency  # plot every how many trials?
    plot_eye_snippet = plot_eye_snippet  # False or an integer number of seconds representing duration
    eye_start_time = df.index[0]
    eye_end_time = df.index[-1]
    count = 0
    eye_results = []
    class_onsets = []
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
            eye_results.append({})
            class_onsets.append({})
            continue

        if plot_eye_result:
            plot_segments(eye_data, row.ppid, row.session, row.block, row.number_in_block, timestamp_start,
                          timestamp_end=plot_timestamp_end, classifiers=classifiers, save_path=save_path)
        if 'NSLR' in classifiers:  # default to NSLR
            classifier = "NSLR"
        elif 'REMODNAV' in classifiers:
            classifier = "REMODNAV"
        else:
            classifier = None
        if classifier:
            (seg_time, seg_class) = continuous_to_discrete(eye_data['timestamp'],
                                                           eye_data[classifier + "_Segment"],
                                                           eye_data[classifier + "_Class"])
            seg_df = pd.DataFrame([seg_time, seg_class]).T
            seg_df.columns = ['seg_time', 'seg_class']
            class_onsets.append(seg_df.values.tolist())
            seg_df.seg_time = pd.to_numeric(seg_df.seg_time)
            seg_df.loc[:, 'duration'] = seg_df['seg_time'].diff().shift(-1)
            seg_df['duration'].iloc[-1] = eye_data['timestamp'].iloc[-1] - seg_df['seg_time'].iloc[-1]
            seg_df_summary = seg_df[['seg_time', 'seg_class', 'duration']].groupby('seg_class').describe()
            seg_df_summary = pd.DataFrame(
                seg_df_summary[[('seg_time', 'count'), ('seg_time', 'min'), ('duration', 'mean')]])
            seg_df_summary.columns = seg_df_summary.columns.droplevel(0)
            seg_df_summary.columns = [f"{classifier}_count", f"{classifier}_first_onset", f"{classifier}_mean_duration"]
            eye_results.append(seg_df_summary.to_dict())
        count += 1
    eye_results = pd.json_normalize(eye_results)
    eye_results['class_onsets'] = class_onsets
    post_processed_event_df = pd.concat([event_df, eye_results], axis=1)
    return post_processed_event_df
