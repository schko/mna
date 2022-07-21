from data_utils import RNStream
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import json, os
from datetime import datetime
import pandas as pd

# Convert RN app data format to MNE
def process_file(input_file, save_pickle=False, data_dir='./../In-Lab Recordings/', pickle_dir='./../Pkl_Recordings/'):
    print(f"processing file {input_file}")

    def get_RNStream(file_path, ignore_stream=None, only_stream=None, jitter_removal=False):
        rns_object = RNStream(data_dir + file_path)
        rns_data = rns_object.stream_in(ignore_stream=ignore_stream, only_stream=only_stream,
                                        jitter_removal=jitter_removal)
        key_list = list(rns_data.keys())
        for key in key_list:
            new_key = key
            if new_key.isdigit():
                new_key = 'video_' + new_key
            new_key = new_key.replace('.', '_')
            rns_data[new_key] = rns_data.pop(key)

        return rns_data

    def save_pkl(directory, file_name, data):
        with open(directory + file_name + '.pkl', 'wb') as handle:
            pickle.dump(rns_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rns_data = get_RNStream(input_file, jitter_removal=False)
    if save_pickle:
        save_pkl(directory=pickle_dir, file_name=input_file, data=rns_data)  # temporary so we can load it quickly
    return rns_data

# Utility functions to read JSON Data
def read_all_files(data_dir='./../In-Lab Recordings/'):
    """
    Processes all RN files in a directory, returns dictionary representations
    """
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    converted_files = []
    for input_file in onlyfiles:
        converted_files.append(process_file(input_file, data_dir = data_dir))
    return converted_files

def read_all_lslpresets(path_to_jsonfiles = './LSLPresets/'):
    metadata_jsons = []
    for file in os.listdir(path_to_jsonfiles):
        full_filename = "%s/%s" % (path_to_jsonfiles, file)
        with open(full_filename,'r') as fi:
            dict = json.load(fi)
            metadata_jsons.append(dict)
    return metadata_jsons

def return_metadata_from_name(stream_name, metadata_jsons):
    for stream in metadata_jsons:
        if stream['StreamName'] == stream_name or stream['StreamName'] == stream_name.replace("_", "."):
            return stream
    return None

# Event Dataframe

tstodt = lambda a : datetime.fromtimestamp(a)

'''
Experimentally, we have the following order:
- density changes
- about a 2 second break
- beep
- particpant gives voice input
- voice data comes back through LSL in the form of AIYVoice (hence forward fill)
- the current trial density, etc. gets sent through LSL in the form of Unity_TrialInfo
- the current chunk info gets sent through LSL in the form of Unity_ChunkInfo (hence backward fill)
- next trial starts
- some time goes by (?) and inference result back from model in the form of PubSub
'''
# experimentally, we get voice input for a trialwe send Unity_TrialInfo, followed immediately by Unity_ChunkInfo

def add_trial_start_time(event_df, offset=0.01):
    trial_end_times = np.zeros(event_df.shape[0])
    trial_end_times[1:] = event_df.trial_end_time[0:-1]+offset # add a 0.01 second offset since the next trial starts immediately
    event_df.insert(0, "trial_start_time", trial_end_times)

def event_data_from_data(rns_data):
    """
    Takes raw data from LSL streams and merges TrialInfo, ChunkInfo and AIYVoice info into an event dataframe
    """
    event_df = pd.DataFrame(rns_data['Unity_TrialInfo'][0], columns=rns_data['Unity_TrialInfo'][1],
                  index=rns_data['Unity_TrialInfo'][2]['ChannelNames']).T
    # chunk data is always paired but offset
    if 'Unity_ChunkInfo' in rns_data:
        chunk_df = pd.DataFrame(rns_data['Unity_ChunkInfo'][0], columns=rns_data['Unity_ChunkInfo'][1],
                              index=rns_data['Unity_ChunkInfo'][2]['ChannelNames']).T
        chunk_df['chunk_timestamp'] = chunk_df.index
        event_df = pd.merge_asof(event_df, chunk_df,
                                 left_index=True,right_index=True,
                                 direction='nearest', tolerance=1)
    else:
        print(f"Unity_ChunkInfo not found")
    # voice data
    voice_df = pd.DataFrame(rns_data['AIYVoice'][0], columns=rns_data['AIYVoice'][1],
                      index=['spoken_difficulty']).T
    voice_df['voice_timestamp'] = voice_df.index
    # we expect voice data then trial end within 15 seconds
    event_df = pd.merge_asof(event_df, voice_df,
                                 left_index=True,right_index=True,
                                 direction='backward', tolerance=15)
    event_df = event_df.reset_index().rename(columns={'index':'trial_end_time'})
    add_trial_start_time(event_df)
    return event_df
