import mne
import numpy as np
import autoreject
import matplotlib.pyplot as plt
import pandas as pd
import os
from mne_features.univariate import compute_hjorth_mobility,compute_pow_freq_bands
from mne.preprocessing import corrmap
from eeglib.features import (bandPower, hjorthActivity, hjorthMobility,
                             hjorthComplexity, sampEn, LZC, DFA, _HFD, HFD, PFD,
                             synchronizationLikelihood)


def process_session_eeg(rns_data, event_df, event_column='spoken_difficulty_encoded', eeg_channel='BioSemi',
                        eeg_montage='biosemi64', save_path='../output/', sbj_session = None, save_raw_eeg = False,
                        run_autoreject=True, autoreject_epochs=20, run_ica=True, template_ica = None, average_reference=True,
                        downsampling = True, n_decim = 16, low_cut=1, hi_cut=55, plot_epochs=False, bands_limits=None, 
                        analyze_pre_ica = False, eye_movement_removal=True, tmin=-.2, tmax=3, baseline=(None, 0),
                        normalize_pow_freq=True, filter_events=True):

    if bands_limits is None:
        bands_limits = [4, 8, 15, 32, 55]
    if filter_events:
        event_detected = event_df[event_column].notnull()
        event_recognized_df = event_df[event_detected]
    else:
        event_recognized_df = event_df

    eeg_channel_names = mne.channels.make_standard_montage(eeg_montage).ch_names
    df = pd.DataFrame(rns_data[eeg_channel][0], columns=rns_data[eeg_channel][1],
                      index=rns_data[eeg_channel][2]['ChannelNames']).T
    starting_time_s = rns_data[eeg_channel][1][0]
    freq = rns_data[eeg_channel][2]['NominalSamplingRate']
    rna_channel_names = list(df.columns)
    rna_channel_names[1:65] = eeg_channel_names
    info = mne.create_info(ch_names=rna_channel_names, ch_types=['stim'] + ['eeg'] * 64 + ['ecg'] * 2 + ['misc'] * 22,
                           sfreq=freq)
    info.set_montage(eeg_montage)

    raw = mne.io.RawArray(df.T * 1e-6, info)
    raw = raw.pick('eeg')
    raw = raw.set_eeg_reference(ref_channels='average',projection=True)
    if average_reference:
        raw = raw.apply_proj()  # set average reference
    if low_cut or hi_cut:
        raw.filter(l_freq=low_cut, h_freq=hi_cut)
    if downsampling:
        raw.resample(freq/n_decim)
        freq /= n_decim

    trial_start_time = event_recognized_df.trial_start_time - starting_time_s  # reference for mne
    event_values = event_recognized_df[event_column].values
    events = np.column_stack((trial_start_time.values * freq,
                              np.zeros(len(event_recognized_df), dtype=int),
                              event_values)).astype(int)
    event_dict = dict(easy=1, hard=2, unknown=0)
    
    # save raw data
    if save_raw_eeg:
        
        raw_eeg_dir = '../output/saved_files/raw_eeg/'
        if not os.path.isdir(raw_eeg_dir): os.makedirs(raw_eeg_dir)
        raw.save(os.path.join(raw_eeg_dir, sbj_session+'_eeg_filt_raw.fif'), overwrite=True)

    if run_ica:
        
        # create duplicate raw, event_df, event_recognized_df for pre and post ica comparison
        if analyze_pre_ica:
            raw_pre_ica = raw.copy()
            event_df_pre_ica = event_df
            event_recognized_df_pre_ica = event_recognized_df

        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=64, random_state=64) # n_components as a decimal set % explained variance
        ica.fit(raw)
        
        # ica.plot_components(picks = range(0,10))
        
        # # Automatic Artifact Detection
        # eog_idx, eog_scores = ica.find_bads_eog(raw, ch_name = ['Fp1', 'Fp2'])
        
        # Semi automatic artifact detection - Corrmap
        if (sbj_session == 'sbj20ssn03') or (template_ica == None):
            if eye_movement_removal:
                eog_idx = [4, 5]
            else:
                eog_idx = [5]
        else:
            icas = [template_ica]+[ica]
            corrmap(icas, template= (0,5), label = "blink", show=False)
            
            if eye_movement_removal:
                corrmap(icas, template= (0,4), label = "horizontal_eye_movement", show=False)
            identified_ica_label = [ica.labels_ for ica in icas]
            
            if eye_movement_removal:
                eog_idx = identified_ica_label[1]['blink']+identified_ica_label[1]['horizontal_eye_movement']
            else:
                eog_idx = identified_ica_label[1]['blink']
        
        # Reconstruct filtered raw signal without Eye Components
        ica.apply(raw, exclude=eog_idx)

    else:
        ica = None
        eog_idx = None
    
    def process_session_eeg_inner(raw, event_recognized_df, event_df):
        if average_reference:
            epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, 
                                    on_missing='warn', metadata=event_recognized_df)
        else:
            epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, proj='delayed',
                                    on_missing='warn', metadata=event_recognized_df)
        event_recognized_df = event_recognized_df[[e==() for e in epochs.drop_log]] # only keep good epochs in event_df
        reject_log = None
        
        # EEG Feature Extraction - 24 features
        extracted_24_features_df = eeg_features(epochs, event_recognized_df, bands_limits, eeg_channel_names, freq, normalize_pow_freq=normalize_pow_freq)

        if len(epochs) < 10: # we need at least 10 epochs to run autoreject for cross validation
            bad_epochs = pd.Series(np.full(len(event_df),np.NAN), index=event_df.index, name='autorejected')
            event_df = event_df.join(bad_epochs)
            reject_log = None
        elif run_autoreject:
            ar = autoreject.AutoReject(random_state=11,
                                       n_jobs=1, verbose=False)
            ar.fit(epochs[:autoreject_epochs])  # fit on a few epochs to save time
            epochs_ar, reject_log = ar.transform(epochs, return_log=True)
            bad_epochs = pd.Series(reject_log.bad_epochs, index=event_recognized_df.index, dtype=bool, name='autorejected')
            event_df = event_df.join(bad_epochs)
            epochs = epochs_ar
            
        return epochs, event_recognized_df, reject_log, event_df, extracted_24_features_df
    
    epochs, event_recognized_df, reject_log, event_df, extracted_24_features_df = process_session_eeg_inner(raw,
                                                                                    event_recognized_df, event_df)
        
    try:
        if plot_epochs:
            if not os.path.isdir(save_path): os.makedirs(save_path)
            ppid = event_recognized_df.iloc[0].ppid
            session = event_recognized_df.iloc[0].session
            block = event_recognized_df.iloc[0].block
            trial = event_recognized_df.iloc[0].number_in_block

            easy = epochs['easy'].average()
            hard = epochs['hard'].average()
            fig, axd = plt.subplot_mosaic([['left', 'left', 'right', 'right'],
                                           ['lower left left', 'lower left right', 'lower right left',
                                            'lower right right']],
                                          figsize=(15, 12), constrained_layout=True)

            easy.plot(spatial_colors=True, axes=axd['left'])
            hard.plot(spatial_colors=True, axes=axd['right'])
            epochs['easy'].plot_psd_topomap(bands=[(8, 12, 'Alpha'), (12, 30, 'Beta')], ch_type='eeg',
                                            normalize=True,
                                            axes=[axd['lower left left'], axd['lower left right']])
            epochs['hard'].plot_psd_topomap(bands=[(8, 12, 'Alpha'), (12, 30, 'Beta')], ch_type='eeg',
                                            normalize=True,
                                            axes=[axd['lower right left'], axd['lower right right']])
            axd['left'].title.set_text('Average (easy)')
            axd['right'].title.set_text('Average (hard)')
            axd['lower left left'].title.set_text('Alpha (8-12Hz)')
            axd['lower left right'].title.set_text('Beta (12-30Hz) PSD (easy)')
            axd['lower right left'].title.set_text('Alpha (8-12Hz)')
            axd['lower right right'].title.set_text('Beta (12-30Hz) PSD (hard)')
            plt.show()
            fig.savefig(f"{save_path}ppid_{ppid}_session_{session}_block_{block}_trial_{trial}_eeg.png")
            plt.close()
            if reject_log:
                result_fig = reject_log.plot('horizontal')
                result_fig.savefig( 
                    f"{save_path}ppid_{ppid}_session_{session}_block_{block}_trial_{trial}_eeg_autoreject_preica.png")
                plt.close()
            ica.plot_components(show=False)
            plt.savefig(f"{save_path}ppid_{ppid}_session_{session}_block_{block}_trial_{trial}_eeg_ica.png")
            plt.close()
    except Exception as e:
        print('some plotting error', e)
        pass
    
    event_df = event_df.join(extracted_24_features_df)
    
    if run_ica and analyze_pre_ica:
        _, _, _, event_df_pre_ica, extracted_24_features_df_pre_ica = process_session_eeg_inner(raw_pre_ica, 
                                                                        event_recognized_df_pre_ica, event_df_pre_ica)
        
        event_df_pre_ica.columns = event_df_pre_ica.columns.str.replace("autorejected", "autorejected_raw")
        extracted_24_features_df_pre_ica = extracted_24_features_df_pre_ica.add_suffix("_raw")

        event_df = event_df.join(event_df_pre_ica['autorejected_raw'])
        event_df = event_df.join(extracted_24_features_df_pre_ica)

    return event_df, epochs, events, info, reject_log, ica, eog_idx

def eeg_features(epochs, event_recognized_df, bands_limits, eeg_channel_names, fs, win_size = 1024, normalize_pow_freq=True):
    
    # identify available frequency bands
    bands = np.asarray(bands_limits)
    band_intervals = list(zip(bands[:-1], bands[1:]))
    
    #initiate empty matrix for all features
    band_power_all = np.empty([len(epochs), len(eeg_channel_names)*len(band_intervals)])
    hjorth_activity = np.empty([len(epochs), len(eeg_channel_names)])
    hjorth_mobility = np.empty([len(epochs), len(eeg_channel_names)])
    hjorth_complexity = np.empty([len(epochs), len(eeg_channel_names)])
    higuchi_fd = np.empty([len(epochs), len(eeg_channel_names)])
    sample_entropy = np.empty([len(epochs), len(eeg_channel_names)])
    
    # create column list for dataframe (band power column arrangement will be different than others due to its function)
    channel_band_power = [f"{chan_name}_{each_band[0]}-{each_band[1]}_Hz_Power"
                              for chan_name in eeg_channel_names
                                  for each_band in band_intervals]
    band_specific_features_list = ['Hjorth_Activity', 'Hjorth_Mobility', 'Hjorth_Complexity', 
                                   'Higuchi_FD', 'Sample_entropy']
    band_specific_features = [f"{chan_name}_{each_band[0]}-{each_band[1]}_Hz_{each_feature}"
                          for each_feature in band_specific_features_list
                              for each_band in band_intervals
                                  for chan_name in eeg_channel_names]
    # combine column name for all features
    all_features = channel_band_power + band_specific_features

    # band power calculation
    for i in range(len(epochs)):
        eeg_data = np.squeeze(epochs.get_data(item=i))
        # Welch's Method
        # band_power = compute_pow_freq_bands(sfreq=fs, data=eeg_data, freq_bands=bands, normalize=normalize_pow_freq,
        #                                     psd_params={'welch_n_fft': win_size, 'welch_n_per_seg': win_size})
        # Multitaper Method
        band_power = compute_pow_freq_bands(sfreq=fs, data=eeg_data, freq_bands=bands, normalize=normalize_pow_freq,
                                            psd_method = 'multitaper')
        band_power_all[i, :] = band_power
    
    # Other features calculation (hjorth activity, mobility, and complexity; Higuchi FD; Sample entropy)
    for index, freq_band in enumerate(band_intervals):
        
        # filter applies to epochs copy to prevent modifying original epochs
        band_specific_epoch = np.squeeze(epochs.copy().filter(freq_band[0],freq_band[1], verbose = False).get_data())
        
        for i in range(len(epochs.events)):
            for ii in range(64):
                # band-specific Hjorth activity, mobility and complexity
                hjorth_activity[i,ii] = hjorthActivity(band_specific_epoch[i][ii])
                hjorth_mobility[i,ii] = hjorthMobility(band_specific_epoch[i][ii])
                hjorth_complexity[i,ii] = hjorthComplexity(band_specific_epoch[i][ii])

                # band-specific HFD
                higuchi_fd[i,ii] = HFD(band_specific_epoch[i][ii])

                # band-specific sample entropy
                sample_entropy[i,ii] = sampEn(band_specific_epoch[i][ii])
                
    # concatenate each band specific features into a single matrix 
        if index == 0:
            hjorth_activity_all = hjorth_activity
            hjorth_mobility_all = hjorth_mobility
            hjorth_complexity_all = hjorth_complexity
            higuchi_fd_all = higuchi_fd
            sample_entropy_all = sample_entropy
        else:
            hjorth_activity_all = np.hstack((hjorth_activity_all, hjorth_activity))
            hjorth_mobility_all = np.hstack((hjorth_mobility_all, hjorth_mobility))
            hjorth_complexity_all = np.hstack((hjorth_complexity_all, hjorth_complexity))
            higuchi_fd_all = np.hstack((higuchi_fd_all, higuchi_fd))
            sample_entropy_all = np.hstack((sample_entropy_all, sample_entropy))
    
    # concatenate all features into a single matrix and convert to dataframe
    all_eeg_features = np.hstack((band_power_all, hjorth_activity_all, hjorth_mobility_all,
                                  hjorth_complexity_all, higuchi_fd_all, sample_entropy_all)) 
    all_eeg_features_df = pd.DataFrame(all_eeg_features, index=event_recognized_df.index, columns=all_features)

    return all_eeg_features_df