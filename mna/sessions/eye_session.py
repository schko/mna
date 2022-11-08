from mna.modalities.eye_process import plot_segments, continuous_to_discrete, process_eye, clean_and_classify, coords_to_degree
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy

def process_session_eye(rns_data, event_df, eye_channel='Unity_ViveSREyeTracking', detect_blink=True, 
                        pretrial_period=0, posttrial_period=0, plot_frequency=20,
                        plot_eye_snippet=40, save_path='../output/', classifiers=None, plot_eye_result=False):
    """
    event_df: dataframe with timestamps start and end to iterate over
    pretrial_period: how much time before the trial_start to use for classification (secs), shorter time = less accurate 
                    classifications
    plot_frequency: plot how many other trials?
    plot_ecg_snippet: plot how many seconds per plot?
    """

    if classifiers is None:
        classifiers = ['NSLR', 'REMODNAV']

    df = pd.DataFrame(rns_data[eye_channel][0], columns=rns_data[eye_channel][1],
                      index=rns_data[eye_channel][2]['ChannelNames']).T
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    eye_start_time = df.timestamp[0]
    eye_end_time = df.timestamp[-1]
    
    count = 0
    eye_results = defaultdict(list)
    class_onsets = defaultdict(list)
    
    fs = rns_data[eye_channel][2]['NominalSamplingRate']
    
    for index, row in event_df.iterrows():
        if plot_frequency > 0 and count % plot_frequency == 0:
            plot_eye_result = True
        else:
            plot_eye_result = False
        orig_timestamp_start = row['trial_start_time']
        orig_timestamp_end = row['trial_end_time']
        timestamp_start = row['trial_start_time'] - pretrial_period
        timestamp_end = row['trial_end_time'] + posttrial_period

        timestamp_start = max(timestamp_start, eye_start_time)  # in case we don't have data starting from e.g. 0
        timestamp_end = min(timestamp_end, eye_end_time)  # in case the trial marker ends before eye data

        if plot_eye_result:
            plot_timestamp_end = min(timestamp_start + plot_eye_snippet, timestamp_end)  # plot 30 seconds of data
        else:
            plot_timestamp_end = timestamp_end

        eye_data, intervals_nan = process_eye(df, detect_blink=detect_blink, timestamp_start=timestamp_start,
                                              timestamp_end=timestamp_end, eye_channel='L',
                                              classifiers=classifiers)
        for c in classifiers:
            if pretrial_period > 0:
                first_valid_segment = eye_data[eye_data.timestamp>=orig_timestamp_start].iloc[0][c+'_Segment']
                eye_data[c+'_Segment'] = eye_data[c+'_Segment']-first_valid_segment+1
            if posttrial_period > 0:
                last_valid_segment = eye_data[eye_data.timestamp<=orig_timestamp_end].iloc[0][c+'_Segment']
        eye_data = eye_data[(eye_data.timestamp>=orig_timestamp_start) & (eye_data.timestamp<=orig_timestamp_end)]
        if pretrial_period > 0:
            try:
                idx_floor = next(x for x, val in enumerate(intervals_nan)
                                                  if val[0] > orig_timestamp_start)
                intervals_nan[idx_floor][0] = max(orig_timestamp_start,intervals_nan[idx_floor][0])
                intervals_nan = intervals_nan[idx_floor:]
            except StopIteration:
                intervals_nan = []
        if posttrial_period > 0: # untested 
            try:
                idx_ceil = next(x for x, val in enumerate(intervals_nan)
                                                  if val[0] > orig_timestamp_end)
                intervals_nan[idx_ceil-1][1] = max(orig_timestamp_end,intervals_nan[idx_ceil-1][1])
                intervals_nan = intervals_nan[:idx_ceil]
            except StopIteration:
                pass
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
            if not seg_df_summary.empty:
                seg_df_summary = pd.DataFrame(
                    seg_df_summary[[('seg_time', 'count'), ('seg_time', 'min'), ('duration', 'mean')]])
                seg_df_summary.columns = seg_df_summary.columns.droplevel(0)
                seg_df_summary.columns = [f"{classifier}_count", f"{classifier}_first_onset", f"{classifier}_mean_duration"]
                eye_results[classifier].append(seg_df_summary.to_dict())
            else:
                eye_results[classifier].append({f"{classifier}_count": None, f"{classifier}_first_onset": None, f"{classifier}_mean_duration": None})
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
        L_Pupil_Diameter_avg = (df['L Pupil Diameter'][(df.timestamp >= event_df['trial_start_time'][index]) &
                                             (df.timestamp <= event_df['trial_end_time'][index]) &
                                                       (df['L Pupil Diameter'] > -1)]).replace(-1, np.nan).mean()
        R_Pupil_Diameter_avg = (df['R Pupil Diameter'][(df.timestamp >= event_df['trial_start_time'][index]) &
                                             (df.timestamp <= event_df['trial_end_time'][index]) &
                                                       (df['R Pupil Diameter'] > -1)]).replace(-1, np.nan).mean()
        L_Pupil_Diameter_trial.append(L_Pupil_Diameter_avg)
        R_Pupil_Diameter_trial.append(R_Pupil_Diameter_avg)

    L_Pupil_Diameter_trial_avg = pd.DataFrame(L_Pupil_Diameter_trial, columns=["Left Pupil Diameter"])
    R_Pupil_Diameter_trial_avg = pd.DataFrame(R_Pupil_Diameter_trial, columns=["Right Pupil Diameter"])
    
    L_Pupil_3sec_evoked = pd.DataFrame(pupil_diameter_evoked(df, event_df, fs, 'L Pupil Diameter'), 
                                 columns=["Left Evoked Pupil Diameter"])
    R_Pupil__3sec_evoked = pd.DataFrame(pupil_diameter_evoked(df, event_df, fs, 'R Pupil Diameter'), 
                                 columns=["Right Evoked Pupil Diameter"])

    post_processed_event_df = pd.concat([post_processed_event_df,
                                        L_Pupil_Diameter_trial_avg, R_Pupil_Diameter_trial_avg,
                                        L_Pupil_3sec_evoked, R_Pupil__3sec_evoked], axis = 1)

    return post_processed_event_df

def process_eye_trial_xlsx(xlsx_filename, save_path='../output/', classifiers='NSLR', viewing_dist=65,
                           screen_max_x=1280, screen_max_y=960, plot_eye_result = False, time_units='s',
                           pupil_channel='pupil', start_timestamp=None):
    """
    A session in this case is actually a full trial.
    """
    if isinstance(classifiers, str):
        classifiers = [classifiers]
    fig = None
    wilming_data = pd.read_excel(xlsx_filename)
    if not start_timestamp:
        start_timestamp = 0
    wilming_data = wilming_data.loc[wilming_data.time >= start_timestamp]
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

def pupil_diameter_evoked(eyetracking_df, event_df, fs, pupil, plot_evoked_pupil = False):
    
    time = np.arange(0, 3, 1/fs)
    trial_onset = event_df['trial_start_time']
    baseline_samples = int(fs*0.2)
    onset_3sec_samples = int(fs*3)
    
    baseline_pupil = np.empty(len(trial_onset.index))
    onset_3sec_pupil = np.empty([len(trial_onset.index),onset_3sec_samples])

    for i in event_df.index:
        
        # filter first trial and trial exceed available eyetracking timestamp
        if (trial_onset[i] == 0) or (trial_onset[i] > max(eyetracking_df.timestamp)): 
            baseline_pupil[i] = float("NaN")
            onset_3sec_pupil[i,:] = float("NaN")
        else:
            trial_baseline_pupil = eyetracking_df[pupil][(eyetracking_df.timestamp >= trial_onset[i]-.2) & 
                                                         (eyetracking_df.timestamp < trial_onset[i])].replace(-1, np.nan)
            
            trial_onset_3sec_pupil = eyetracking_df[pupil][(eyetracking_df.timestamp >= trial_onset[i]) & 
                                                        (eyetracking_df.timestamp < trial_onset[i]+3)].replace(-1, np.nan)
            
            # fill in missing values through interpolation
            trial_baseline_pupil = trial_baseline_pupil.interpolate(method ='linear', 
                                                                    limit_direction ='forward')
            trial_onset_3sec_pupil = trial_onset_3sec_pupil.interpolate(method ='linear', 
                                                                        limit_direction ='forward')

            # keep all trial dimension consistent through linear interpolation (due to inconsistent sample rate)
            if len(trial_baseline_pupil) != baseline_samples:
                baseline_interp = scipy.interpolate.interp1d(np.arange(trial_baseline_pupil.size),
                                                             trial_baseline_pupil)
                trial_baseline_pupil_interp = baseline_interp(np.linspace(0, trial_baseline_pupil.size-1, baseline_samples))
            else: 
                trial_baseline_pupil_interp = trial_baseline_pupil

            if len(trial_onset_3sec_pupil) != onset_3sec_samples:
                onset_3sec_pupil_interp = scipy.interpolate.interp1d(np.arange(trial_onset_3sec_pupil.size),
                                                                     trial_onset_3sec_pupil)
                trial_onset_3sec_pupil_interp = onset_3sec_pupil_interp(np.linspace(0, trial_onset_3sec_pupil.size-1, onset_3sec_samples))
            else: 
                trial_onset_3sec_pupil_interp = trial_onset_3sec_pupil

            # invalidate trials with more than half invalid samples
            if (np.isnan(trial_baseline_pupil_interp).sum() >= int(baseline_samples*.5) or 
                np.isnan(trial_onset_3sec_pupil_interp).sum() >= int(onset_3sec_samples*.5)):
                baseline_pupil[i] = float("NaN")
            else:
                baseline_pupil[i] = trial_baseline_pupil_interp.mean()
        
            # baseline correction
            onset_3sec_pupil[i,:] = trial_onset_3sec_pupil_interp - baseline_pupil[i]
    
    # evoked pupil diameter
    evoked_pupil_diameter = np.mean(onset_3sec_pupil, axis = 1)
    
    # plot easy and hard trials - averaged
    valid_baseline_trials = onset_3sec_pupil[~np.isnan(baseline_pupil)]
    easy_trials = onset_3sec_pupil[event_df['spoken_difficulty']=='easy']
    hard_trials = onset_3sec_pupil[event_df['spoken_difficulty']=='hard']

    def plot_evoked_trials(trials, line_color, fill_color, trial_label):
        trials_mean = np.nanmean(trials, axis = 0)
        trials_std = np.nanstd(trials, axis = 0)

        under_line = trials_mean - trials_std
        over_line = trials_mean + trials_std

        plt.plot(time, trials_mean, line_color, label = trial_label)
        plt.fill_between(time, under_line, over_line, color=fill_color, alpha=.1)
        plt.legend()
        
    if plot_evoked_pupil:
        plot_evoked_trials(easy_trials, 'r-', 'r', 'Easy')
        plot_evoked_trials(hard_trials, 'k-', 'k', 'Hard')
    
    return evoked_pupil_diameter