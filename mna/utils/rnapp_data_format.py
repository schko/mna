from os import listdir
from os.path import isfile, join
import pickle
import json
from datetime import datetime
import pandas as pd
import os
import numpy as np
import warnings
from multiprocessing import Pool 


# constant
magic = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
max_label_len = 32
max_dtype_len = 8
dim_bytes_len = 8
shape_bytes_len = 8

endianness = 'little'
encoding = 'utf-8'

ts_dtype = 'float64'

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
    if not os.path.isfile(pickle_dir + input_file + '.pkl'):
        rns_data = get_RNStream(input_file, jitter_removal=False)
        if save_pickle:
            save_pkl(directory=pickle_dir, file_name=input_file, data=rns_data)  # temporary so we can load it quickly
    return rns_data

def multi_run_wrapper(args):
   return process_file(*args)

# Utility functions to read JSON Data
def read_all_files(data_dir='./../In-Lab Recordings/', pickle_dir='./../Pkl_Recordings/', save_pickle=False):
    """
    Processes all RN files in a directory, returns dictionary representations
    """
    onlyfiles = [(f, True, data_dir, pickle_dir) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    pool_obj = Pool(12)
    converted_files = pool_obj.map(multi_run_wrapper, onlyfiles)
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

def add_trial_start_time(event_df, first_trial_start_time=0, offset=0.001):
    trial_end_times = np.zeros(event_df.shape[0])
    trial_end_times[1:] = event_df.trial_end_time[0:-1]+offset # add a 0.01 second offset since the next trial starts immediately
    event_df.insert(0, "trial_start_time", trial_end_times)
    event_df.trial_start_time[0] = first_trial_start_time
    
def read_event_data(rns_data,  continuous_channel='Unity_MotorInput', remove_id_sessions=[(15, 1), (22, 1)],
                         trial_df_path='../data/annotated/', override_ppid_sess = False):
    """
    Takes directory of trial_dfs.
    Removed_ids = 15-1 and 22-1 incomplete sessions where participant got sick
    """
    first_trial_start_time = rns_data[continuous_channel][1][0]
    if not override_ppid_sess:
        tmp_event_df = pd.DataFrame(rns_data['Unity_TrialInfo'][0], columns=rns_data['Unity_TrialInfo'][1],
                                index=rns_data['Unity_TrialInfo'][2]['ChannelNames']).T
        ppid = tmp_event_df.iloc[0].ppid
        session = tmp_event_df.iloc[0].session
    else:
        ppid = override_ppid_sess[0]
        session = override_ppid_sess[1]

    removed_ids = [p[0] for p in remove_id_sessions]
    removed_sessions = [p[1] for p in remove_id_sessions]

    if ppid in removed_ids and session in removed_sessions:
        return pd.DataFrame()

    event_df = pd.read_csv(f"{trial_df_path}{int(ppid)}_{int(session)}_trial_df.csv")
    event_df = event_df.rename(columns={'timestamps':'trial_end_time', 'voice_response': 'spoken_difficulty'})
    add_trial_start_time(event_df, first_trial_start_time)
    event_df.insert(1, 'trial_end_time', event_df.pop('trial_end_time'))
    return event_df

def event_data_from_data(rns_data, ts_fixer, adjust_event_ts = True, adjust_voice_ts = False, ensure_minimum_trial = False, continuous_channel = 'Unity_MotorInput', remove_id_sessions=[(13,1), (15,1), (22,1)], interrupted_id_sessions=[(13,1), (22,1)], offset_parameters=None):
    """
    Takes raw data from LSL streams and merges TrialInfo, ChunkInfo and AIYVoice info into an event dataframe.
    Applies timestamp fixer (tsfixer) to adjust timestamps.
    Removed_ids = 13-1 and 22-1: interrupted sessions, no detectable calibration periods nor regular calibration intervals; 15-1 and 22-1 incomplete sessions where participant got sick
    """
    first_trial_start_time = rns_data[continuous_channel][1][0]

    event_df = pd.DataFrame(rns_data['Unity_TrialInfo'][0], columns=rns_data['Unity_TrialInfo'][1],
                  index=rns_data['Unity_TrialInfo'][2]['ChannelNames']).T
    event_df = event_df.reset_index().rename(columns={'index': 'trial_end_time'})
    
    removed_ids = [p[0] for p in remove_id_sessions]
    removed_sessions = [p[1] for p in remove_id_sessions]
    
    interrupted_ids = [p[0] for p in interrupted_id_sessions]
    interrupted_sessions = [p[1] for p in interrupted_id_sessions]

    if event_df.iloc[0].ppid in removed_ids and event_df.iloc[0].session in removed_sessions:
        return pd.DataFrame()
        
    # chunk data is always paired but offset
    if 'Unity_ChunkInfo' in rns_data:
        chunk_df = pd.DataFrame(rns_data['Unity_ChunkInfo'][0], columns=rns_data['Unity_ChunkInfo'][1],
                              index=rns_data['Unity_ChunkInfo'][2]['ChannelNames']).T
        chunk_df = chunk_df.reset_index().rename(columns={'index': 'chunk_timestamp'})
        event_df = pd.concat([event_df,chunk_df],axis=1)
        #chunk_df['chunk_timestamp'] = chunk_df.index
        #event_df = pd.merge_asof(event_df, chunk_df,
        #                         left_on="trial_start_time",right_index=True,
        #                         direction='nearest', tolerance=1)
    else:
        print(f"Unity_ChunkInfo not found")
    
    if event_df.iloc[0].ppid in interrupted_ids and event_df.iloc[0].session in interrupted_sessions:
        print('FIXING THE EVENT DF SINCE PID', event_df.iloc[0].ppid, 'SESSION', event_df.iloc[0].session, 'WAS INTERRUPTED')
        event_df = event_df.loc[~event_df.duplicated(subset=['ppid','session','block','number_in_block','trial'], keep='first'),:].reset_index(drop=True)
        last_freak_idx = event_df.loc[event_df.ppid == 0].index[-1]
        event_df = event_df[event_df.ppid != 0].reset_index(drop=True)
        event_df.loc[last_freak_idx:,'session'] = event_df.loc[last_freak_idx-1,'session']
        event_df.loc[last_freak_idx:,'block'] = event_df.loc[last_freak_idx:,'block'] + event_df.loc[last_freak_idx-1,'block']
        event_df.loc[last_freak_idx:,'trial'] = event_df.loc[last_freak_idx:,'trial'] + event_df.loc[last_freak_idx-1,'trial']
        event_df.loc[last_freak_idx:,'damage'] = event_df.loc[last_freak_idx:,'damage'] + event_df.loc[last_freak_idx-1,'damage']
    event_df['non_adj_trial_end_time'] = event_df.trial_end_time
    if adjust_event_ts:
        feats = pd.DataFrame([event_df.trial_end_time.diff(),
                              event_df.trial_end_time.diff().shift(1),
                              event_df.trial_end_time.diff().shift(-1),
                              event_df.trial_end_time.diff().shift(-2)]).T
        nan_mask = feats.isna().any(axis=1)
        ref_start_trial_end = event_df.loc[~nan_mask,'trial_end_time'].iloc[0]
        first_valid_index = event_df.loc[~nan_mask].index[0]
        last_valid_index = event_df.loc[~nan_mask].index[-1]
        feats_na = feats.dropna()
        pred_offsets = ts_fixer.predict(feats_na.values)
        cumulative_time_offset = pred_offsets.cumsum()
        event_df.iloc[first_valid_index:last_valid_index+1,event_df.columns.get_loc("trial_end_time")] = (np.array(cumulative_time_offset)+ref_start_trial_end)
        # ensure a minimum possible trial
        if ensure_minimum_trial:
            trial_end_times = event_df.trial_end_time.copy()
            trial_dur_threshold = 6 # need at least a minimum of this, realistic, duration
            for index, row in event_df[1:].iterrows():
                trial_duration = row.trial_end_time-event_df.iloc[index-1].trial_end_time
                if trial_duration <= trial_dur_threshold:
                    new_trial_duration = np.random.normal(trial_dur_threshold+1,1)
                    adjustment = new_trial_duration - trial_duration
                    trial_end_times[index:] = trial_end_times[index:] + adjustment
            event_df.trial_end_time = trial_end_times
        event_df = event_df.iloc[:last_valid_index+1]

    add_trial_start_time(event_df,first_trial_start_time)
    
    # voice data
    voice_df = pd.DataFrame(rns_data['AIYVoice'][0], columns=rns_data['AIYVoice'][1],
                      index=['spoken_difficulty']).T
    voice_df = voice_df.reset_index().rename(columns={'index': 'voice_timestamp'})
    voice_df['non_adj_voice_timestamp'] = voice_df.voice_timestamp
    if adjust_voice_ts:
        feats = pd.DataFrame([voice_df.voice_timestamp.diff(),
                              voice_df.voice_timestamp.diff().shift(1),
                              voice_df.voice_timestamp.diff().shift(-1),
                              voice_df.voice_timestamp.diff().shift(-2)]).T
        nan_mask = feats.isna().any(axis=1)
        ref_start_trial_end = voice_df.loc[~nan_mask,'voice_timestamp'].iloc[0]
        first_valid_index = voice_df.loc[~nan_mask].index[0]
        last_valid_index = voice_df.loc[~nan_mask].index[-1]
        feats_na = feats.dropna()
        pred_offsets = ts_fixer.predict(feats_na.values)
        cumulative_time_offset = pred_offsets.cumsum()
        voice_df.iloc[first_valid_index:last_valid_index+1,event_df.columns.get_loc("voice_timestamp")] = (np.array(cumulative_time_offset)+ref_start_trial_end)
        # ensure a minimum possible trial
        if ensure_minimum_trial:
            voice_timestamps = voice_df.voice_timestamp.copy()
            trial_dur_threshold = 6 # need at least a minimum of this, realistic, duration
            for index, row in voice_df[1:].iterrows():
                trial_duration = row.voice_timestamp-voice_df.iloc[index-1].voice_timestamp
                if trial_duration <= trial_dur_threshold:
                    new_trial_duration = np.random.normal(trial_dur_threshold+1,1)
                    adjustment = new_trial_duration - trial_duration
                    voice_timestamps[index:] = voice_timestamps[index:] + adjustment
            voice_df.voice_timestamp = voice_timestamps
        voice_df = voice_df.iloc[:last_valid_index+1]
    
    for index, row in voice_df.iterrows(): # keep the last voice data for each trial
        event_df.loc[((event_df.trial_start_time < row.voice_timestamp) & (event_df.trial_end_time >= row.voice_timestamp)),'spoken_difficulty'] =\
            row['spoken_difficulty']
        event_df.loc[((event_df.trial_start_time < row.voice_timestamp) & (event_df.trial_end_time >= row.voice_timestamp)), 'voice_timestamp'] = \
            row.voice_timestamp
        event_df.loc[((event_df.trial_start_time < row.voice_timestamp) & (event_df.trial_end_time >= row.voice_timestamp)), 'non_adj_voice_timestamp'] = \
            row.non_adj_voice_timestamp
    return event_df

def window_slice(data, window_size, stride, channel_mode='channel_last'):
    assert len(data.shape) == 2
    if channel_mode == 'channel_first':
        data = np.transpose(data)
    elif channel_mode == 'channel_last':
        pass
    else:
        raise Exception('Unsupported channel mode')
    assert window_size <= len(data)
    assert stride > 0
    rtn = np.expand_dims(data, axis=0) if window_size == len(data) else []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)

class RNStream:
    def __init__(self, file_path):
        self.fn = file_path

    def stream_in(self, ignore_stream=None, only_stream=None, jitter_removal=True):
        """
        different from LSL XDF importer, this jitter removal assumes no interruption in the data
        :param ignore_stream:
        :param only_stream:
        :param jitter_removal:
        :return:
        """
        total_bytes = float(os.path.getsize(self.fn))  # use floats to avoid scalar type overflow
        buffer = {}
        read_bytes_count = 0.
        with open(self.fn, "rb") as file:
            while True:
                if total_bytes and (100 * read_bytes_count/total_bytes) % 30 == 0:
                #if total_bytes:
                    print('Streaming in progress {0}%'.format(str(round(100 * read_bytes_count/total_bytes, 2))), sep=' ', end='\r', flush=True)
                # read magic
                read_bytes = file.read(len(magic))
                read_bytes_count += len(read_bytes)
                if len(read_bytes) == 0:
                    break
                try:
                    assert read_bytes == magic
                except AssertionError:
                    raise Exception('Data invalid, magic sequence not found')
                # read stream_label
                read_bytes = file.read(max_label_len)
                read_bytes_count += len(read_bytes)
                stream_name = str(read_bytes, encoding).strip(' ')
                # read read_bytes
                read_bytes = file.read(max_dtype_len)
                read_bytes_count += len(read_bytes)
                stream_dytpe = str(read_bytes, encoding).strip(' ')
                # read number of dimensions
                read_bytes = file.read(dim_bytes_len)
                read_bytes_count += len(read_bytes)
                dims = int.from_bytes(read_bytes, 'little')
                # read number of np shape
                shape = []
                for i in range(dims):
                    read_bytes = file.read(shape_bytes_len)
                    read_bytes_count += len(read_bytes)
                    shape.append(int.from_bytes(read_bytes, 'little'))

                data_array_num_bytes = np.prod(shape) * np.dtype(stream_dytpe).itemsize
                timestamp_array_num_bytes = shape[-1] * np.dtype(ts_dtype).itemsize

                this_in_only_stream = (stream_name in only_stream) if only_stream else True
                not_ignore_this_stream = (stream_name not in ignore_stream) if ignore_stream else True
                if not_ignore_this_stream and this_in_only_stream:
                    # read data array
                    read_bytes = file.read(data_array_num_bytes)
                    read_bytes_count += len(read_bytes)
                    data_array = np.frombuffer(read_bytes, dtype=stream_dytpe)
                    data_array = np.reshape(data_array, newshape=shape)
                    # read timestamp array
                    read_bytes = file.read(timestamp_array_num_bytes)
                    ts_array = np.frombuffer(read_bytes, dtype=ts_dtype)

                    if stream_name not in buffer.keys():
                        buffer[stream_name] = [np.empty(shape=tuple(shape[:-1]) + (0,), dtype=stream_dytpe),
                                                                np.empty(shape=(0,))]  # data first, timestamps second
                    buffer[stream_name][0] = np.concatenate([buffer[stream_name][0], data_array], axis=-1)
                    buffer[stream_name][1] = np.concatenate([buffer[stream_name][1], ts_array])
                else:
                    file.read(data_array_num_bytes + timestamp_array_num_bytes)
                    read_bytes_count += data_array_num_bytes + timestamp_array_num_bytes
        if jitter_removal:
            i = 1
            for stream_name, (d_array, ts_array) in buffer.items():
                if len(ts_array) < 2:
                    print("Ignore jitter remove for stream {0}, because it has fewer than two samples".format(stream_name))
                    continue
                if np.std(ts_array) > 0.1:
                    warnings.warn("Stream {0} may have a irregular sampling rate with std {0}. Jitter removal should not be applied to irregularly sampled streams.".format(stream_name, np.std(ts_array)), RuntimeWarning)
                print('Removing jitter for streams {0}/{1}'.format(i, len(buffer)), sep=' ',
                      end='\r', flush=True)
                coefs = np.polyfit(list(range(len(ts_array))), ts_array, 1)
                smoothed_ts_array = np.array([i * coefs[0] + coefs[1] for i in range(len(ts_array))])
                buffer[stream_name][1] = smoothed_ts_array

        print("Stream-in completed: {0}".format(self.fn))
        return buffer