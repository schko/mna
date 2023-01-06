from mna.utils.rnapp_data_format import read_all_lslpresets, return_metadata_from_name, event_data_from_data
import pickle, os
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display

import matplotlib.pyplot as plt
from mna.sessions.eye_session import process_session_eye
from mna.sessions.eeg_session import process_session_eeg
from mna.sessions.motor_session import process_session_motor
from mna.sessions.ecg_session import process_session_ecg
from os import listdir
from os.path import isfile, join
from mna.utils.rnapp_data_format import read_all_lslpresets, return_metadata_from_name, event_data_from_data
import pickle
# 1. Read a RN App, converted pkl file, and create the metadata and data structure


def batch_feature_extraction(data_dir = "data/", lsl_dir = "mna/LSLPresets/", output_dir = 'output/', csv_save = 'all_results.csv',
                            xlsx_save = 'all_results.xlsx', pivot_save = 'all_results.html'):
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    metadata_jsons = read_all_lslpresets(path_to_jsonfiles=lsl_dir)
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and '.pkl' in f]

    save_data_pkl = False # save data into pickle files
    save_ica_plts = False # save ICA components plots
    rs = 64 # random seed


    all_dfs = None

    # ica_epochs_dict = {}
    # fddica_dict = {}
    # eog_idx_dict = {}
    # events_dict = {}

    for each_file in onlyfiles:
        input_path = data_dir + each_file

        sbj_id = each_file[each_file.find('Sbj_')+4:each_file.find('-Ssn')]
        ssn_no = each_file[each_file.find('Ssn_')+4:each_file.find('.dats')]

        if len(sbj_id) < 2: sbj = "sbj0"+sbj_id
        else: sbj = "sbj"+sbj_id
        if len(ssn_no) < 2: ssn = "ssn0"+ssn_no
        else: ssn = "ssn"+ssn_no

        with open(input_path, 'rb') as handle:
            rns_data = pickle.load(handle)

        ## Add metadata to data

        for key in rns_data.keys():
            rns_data[key].append(return_metadata_from_name(key, metadata_jsons))

        event_df = event_data_from_data(rns_data)
        event_df['trial_damage'] = event_df.damage.diff().fillna(0)
        event_df['trial_duration'] = event_df.trial_end_time - event_df.trial_start_time
        percent_missing = event_df.notnull().sum() / len(event_df)
        summary_statistics = {}
        summary_statistics['voice_success_rate'] = percent_missing['voice_timestamp']
        if 'chunk_timestamp' in percent_missing:
            summary_statistics['chunk_success_rate'] = percent_missing['chunk_timestamp']
        else:
            summary_statistics['chunk_success_rate'] = 0

        # temporary fix for pilot phase where we had some incomplete data
        if 'block_condition' not in event_df:
            event_df['block_condition'] = 'practice'
            event_df.loc[5:,'block_condition'] = 'voice'

        event_df['spoken_difficulty_encoded'] = event_df.spoken_difficulty.replace(to_replace=['easy', 'hard', 'unknown'],
                                                                              value=[1, 2, None])

        # ecg
        post_processed_event_df = process_session_ecg(rns_data, event_df,plot_frequency=20,plot_ecg_snippet=40)

        # eye
        if 'Unity_ViveSREyeTracking' in rns_data:
            post_processed_event_df = process_session_eye(rns_data, post_processed_event_df,detect_blink=True,plot_frequency=20, plot_eye_snippet=40)

        # eeg
        post_processed_event_df, epochs, events, event_dict, info, reject_log, ica, eog_idx= process_session_eeg(rns_data, post_processed_event_df,
                                                                        event_column='spoken_difficulty_encoded', run_autoreject=True, run_ica=False,
                                                                        raw_eeg_fname = sbj+ssn+"eeg_filt_raw", save_raw_eeg = False, template_ica = None)

        # # motor
        # post_processed_event_df = process_session_motor(rns_data, post_processed_event_df, motor_channel='Unity_MotorInput',
        #                                             plot_motor_result = True, plot_motor_snippet = 30, plot_frequency = 10)


        # save
        post_processed_event_df.to_csv(f"{output_dir}ppid_{post_processed_event_df.iloc[0].ppid}_session_{post_processed_event_df.iloc[0].session}.csv")
        if not type(all_dfs)==pd.core.frame.DataFrame:
            all_dfs = post_processed_event_df
        else:
            all_dfs = pd.concat([all_dfs, post_processed_event_df], ignore_index=True)

    from pivottablejs import pivot_ui
    if csv_save:
        all_dfs.to_csv(f"{output_dir}{csv_save}", index=False)
    if xlsx_save:
        all_dfs.to_excel(f"{output_dir}{xlsx_save}")
    if pivot_save:
        pivot_ui(all_dfs, outfile_path=f"{output_dir}{pivot_save}");
        
def clean_up_adadrive_trials(all_dfs):
    # remove practice trials
    all_dfs_final = all_dfs.copy()
    all_dfs_final = all_dfs_final[all_dfs_final.block_condition!='practice']
    for col in ['ppid','session','block','number_in_block','trial']:
        all_dfs_final[col] = all_dfs_final[col].astype(int)
    all_dfs_final['ppid_session'] = all_dfs_final['ppid'].astype(str) + '_' + \
                                                + all_dfs_final['session'].astype(str)

    all_dfs_final.columns = all_dfs_final.columns.str.replace('.','_')
    all_dfs_final.columns = all_dfs_final.columns.str.replace('measures_','')
    # all_dfs_final.columns = all_dfs_final.columns.str.replace("Band_Power", "Power")
    
    # # rename columns
    # all_dfs_final = all_dfs_final.rename(columns={"Left Pupil Trial Average Diameter": "L Pupil Diameter"})

    # remove trials that are too long
    all_dfs_final = all_dfs_final[all_dfs_final.trial_duration <= 20]
    
    # ECG outlier detection
    for c in ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2']:
        all_dfs_final.loc[all_dfs_final.bpm < 50, c] = np.nan
        all_dfs_final.loc[all_dfs_final.bpm > 200, c] = np.nan
    
    return all_dfs_final