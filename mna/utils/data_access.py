import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
import autoreject
import pickle
import mne
import glob
import re

# loop over the list of csv files
def read_motor_csvs(output_dir):
    print('reading participant-level motor data')
    csv_files = glob.glob(os.path.join(output_dir, "ppid*_motor.csv"))
    all_dfs = None
    for f in csv_files:
        # read the csv file
        if not type(all_dfs) == pd.core.frame.DataFrame:
            all_dfs = pd.read_csv(f)
        else:
            all_dfs = pd.concat([all_dfs, pd.read_csv(f)], ignore_index=True)
    all_dfs = all_dfs[all_dfs.columns.drop(list(all_dfs.filter(regex='Unnamed')))]
    return all_dfs


def get_motor_epochs(output_dir):
    if os.path.isfile(f"{output_dir}cleaned_motor_epochs.pickle"):
        print('found cleaned epochs')
        cleaned_motor_epochs = pickle.load(open(f"{output_dir}cleaned_motor_epochs.pickle", 'rb'))
        return cleaned_motor_epochs
    else:
        epochs_files = glob.glob(os.path.join(output_dir, "**/*ica_epochs.pickle"), recursive=True)
        motor_epochs = []
        for each_file in epochs_files:
            motor_epochs.append(pickle.load(open(each_file, 'rb')))
        motor_epochs = mne.concatenate_epochs(motor_epochs)
        for col in ['ppid', 'session', 'block', 'number_in_block', 'trial']:
            motor_epochs.metadata[col] = motor_epochs.metadata[col].astype(int)
        cleaned_motor_epochs, cleaned_motor_epochs_reject_log = autoreject.AutoReject(random_state=100).fit(
            motor_epochs[::100]).transform(motor_epochs, return_log=True)
        with open(f"{output_dir}cleaned_motor_epochs.pickle", 'wb') as handle_ica:
            pickle.dump(cleaned_motor_epochs, handle_ica, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{output_dir}cleaned_motor_epochs_reject_log.pickle", 'wb') as handle_ica:
            pickle.dump(cleaned_motor_epochs_reject_log, handle_ica, protocol=pickle.HIGHEST_PROTOCOL)
    for col in ['ppid', 'session', 'block', 'number_in_block', 'trial']:
        motor_epochs.metadata[col] = motor_epochs.metadata[col].astype(int)
    return motor_epochs


def get_exposure_epochs(exposure_epochs_file):
    exposure_epochs = pickle.load(open(exposure_epochs_file, 'rb'))
    exposure_epochs = mne.concatenate_epochs(exposure_epochs)
    return exposure_epochs


def get_motor_intensity_info(input_df):
    try:
        all_steer_events = input_df['post_steer_event_raw']
        all_steer_events_finalized = all_steer_events.apply(str_list_to_list)
    except:
        all_steer_events_finalized = input_df['post_steer_event_raw']
    final_pos = lambda final_wheel_pos: np.asarray(final_wheel_pos[-1]) - np.asarray(final_wheel_pos[0])

    input_df["Steer_Wheel_Degree"] = abs(all_steer_events_finalized.apply(final_pos))
    all_dfs = []
    for sub in input_df.ppid.unique():
        sub_df = input_df[input_df.ppid == sub]
        sub_df["Steer_Wheel_Degree_Categorical"] = pd.qcut(sub_df.Steer_Wheel_Degree, 2,
                                                           labels=["Low", "High"])  # 2=High, 1 =Low
        sub_df["Steer_Wheel_Degree_Encoded"] = sub_df.Steer_Wheel_Degree_Categorical.replace({'High': 2, 'Low': 1})
        all_dfs.append(sub_df)
    return pd.concat(all_dfs).reset_index(drop=True)


def get_forward_results(output_dir, forward_type, motor_sensor, inverse_operator, fwd, lambda2):
    # note this crashes if you run more than one type
    if os.path.isfile(f"{output_dir}{forward_type}_motor_forward.pickle"):
        motor_forward = pickle.load(open(f"{output_dir}{forward_type}_motor_forward.pickle", 'rb'))
        return motor_forward
    else:
        print('did not find, creating')
        motor_forward = []
        stc = mne.minimum_norm.apply_inverse_epochs(motor_sensor, inverse_operator,
                                                    lambda2=lambda2, verbose=False,
                                                    method="eLORETA")
        for s in tqdm_notebook(stc):
            motor_forward.append(mne.apply_forward(fwd, s, motor_sensor.info))

        with open(f"{output_dir}low_motor_forward.pickle", 'wb') as handle_ica:
            pickle.dump(motor_forward, handle_ica, protocol=pickle.HIGHEST_PROTOCOL)
        return motor_forward


def str_list_to_list(lst):
    str_no_brackets = re.sub("[\[\]]", "", lst)
    return [float(n) for n in str_no_brackets.split()]