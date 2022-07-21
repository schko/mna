import pandas as pd
from mna.modalities.ecg_process import process_ecg, plot_ecg
import matplotlib.pyplot as plt
import numpy as np
import os

def process_session_ecg(rns_data, event_df, low_bpm=40, high_bpm=200, save_path='../output/', ecg_channel='BioSemi',
                        plot_frequency=20, plot_ecg_snippet=40):
    """
    event_df: dataframe with timestamps start and end to iterate over
    low_bpm: minimum bpm values as within range
    high_bpm: maximum bpm values accepted as within range
    ecg_channel:
    plot_frequency: plot how many other trials?
    plot_ecg_snippet: plot how many seconds per plot?
    """
    df = pd.DataFrame(rns_data[ecg_channel][0], columns=rns_data[ecg_channel][1],
                      index=rns_data[ecg_channel][2]['ChannelNames']).T
    freq = rns_data[ecg_channel][2]['NominalSamplingRate']
    plot_frequency = plot_frequency  # plot every how many trials?
    plot_ecg_snippet = plot_ecg_snippet  # False or an integer number of seconds representing duration

    ecg_results = []
    ecg_start_time = df.index[0]
    ecg_end_time = df.index[-1]
    count = 0
    for index, row in event_df.iterrows():
        if plot_frequency > 0 and count % plot_frequency == 0:
            plot_ecg_result = True
        else:
            plot_ecg_result = False
        timestamp_start = row['trial_start_time']
        timestamp_end = row['trial_end_time']
        timestamp_start = max(timestamp_start, ecg_start_time)  # in case we don't have data starting from e.g. 0
        timestamp_end = min(timestamp_end, ecg_end_time)  # in case the trial marker ends before ECG data
        if plot_ecg_snippet:
            plot_timestamp_end = min(timestamp_start + plot_ecg_snippet, timestamp_end)  # plot few seconds of data
        else:
            plot_timestamp_end = timestamp_end

        working_data, measures, flipped_signal, failed_to_detect = process_ecg(df, freq, low_bpm=low_bpm,
                                                                               high_bpm=high_bpm,
                                                                               timestamp_start=timestamp_start,
                                                                               timestamp_end=timestamp_end)
        if not failed_to_detect and plot_ecg_result:
            if not os.path.isdir(save_path): os.makedirs(save_path)
            plot_ecg(df, working_data, measures,
                     timestamp_start=timestamp_start, timestamp_end=plot_timestamp_end)
            plt.savefig(
                f"{save_path}ppid_{row.ppid}_session_{row.session}_block_{row.block}_trial_{row.number_in_block}_ecg_{timestamp_start}s_{plot_timestamp_end}.png")
            plt.close()
        if not failed_to_detect:  # catch undetected error
            ecg_results.append({'ecg_timestamp_start': timestamp_start, 'ecg_timestamp_end': timestamp_end,
                                'measures': measures, 'flipped_signal': flipped_signal,
                                'failed_to_detect': failed_to_detect,
                                'removed_beat_rate': len(working_data['removed_beats']) / len(working_data['peaklist']),
                                'valid_r_peaks': list(
                                    set(working_data['peaklist'] / freq) - set(working_data['removed_beats'] / freq))})
        else:
            ecg_results.append({'ecg_timestamp_start': timestamp_start, 'ecg_timestamp_end': timestamp_end,
                                'measures': measures, 'flipped_signal': flipped_signal,
                                'failed_to_detect': failed_to_detect,
                                'removed_beat_rate': np.nan, 'valid_r_peaks': []})
        count += 1
        post_processed_event_df = pd.concat([event_df, pd.json_normalize(ecg_results)], axis=1)
    return post_processed_event_df
