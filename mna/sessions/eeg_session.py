import mne
import numpy as np
import autoreject
import matplotlib.pyplot as plt
import pandas as pd
import os
from mne_features.univariate import compute_pow_freq_bands


def process_session_eeg(rns_data, event_df, event_column='spoken_difficulty_encoded', eeg_channel='BioSemi',
                        eeg_montage='biosemi64', save_path='../output/',
                        run_autoreject=True, autoreject_epochs=20, run_ica=True, average_reference=True, low_cut=0.1,
                        hi_cut=55, plot_epochs=True, bands_limits=None):

    if bands_limits is None:
        bands_limits = [4, 8, 15, 32, 55]
    event_detected = event_df[event_column].notnull()
    event_recognized_df = event_df[event_detected]

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
    if average_reference:
        raw = raw.set_eeg_reference(ref_channels='average')  # set average reference
    if low_cut or hi_cut:
        raw.filter(l_freq=low_cut, h_freq=hi_cut)

    trial_start_time = event_recognized_df.trial_start_time - starting_time_s  # reference for mne
    event_values = event_recognized_df[event_column].values
    events = np.column_stack((trial_start_time.values * freq,
                              np.zeros(len(event_recognized_df), dtype=int),
                              event_values)).astype(int)
    event_dict = dict(easy=1, hard=2)
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=- 0.2, tmax=3, preload=True, on_missing='warn')
    event_recognized_df = event_recognized_df[[e==() for e in epochs.drop_log]] # only keep good epochs in event_df
    reject_log = None
    # Band power calculation
    win_size = 1024
    bands = np.asarray(bands_limits)
    band_intervals = list(zip(bands[:-1], bands[1:]))
    band_power_epochs = np.empty([len(epochs), len(eeg_channel_names)*len(band_intervals)])
    eeg_channel_band_power = [f"{chan_name}_{each_band[0]}-{each_band[1]}_Hz_Band_Power"
                              for chan_name in eeg_channel_names
                              for each_band in band_intervals]

    # Approach with MNE_FEATURES
    for i in range(len(epochs)):
        data_mne = np.squeeze(epochs[i].get_data())
        pow_freq_band = compute_pow_freq_bands(sfreq=freq, data=data_mne, freq_bands=bands, normalize=False,
                                                    psd_params={'welch_n_fft': win_size, 'welch_n_per_seg': win_size})
        band_power_epochs[i, :] = pow_freq_band
    # Approach with SCIPY.SIGNAL.WELCH
    # for i in range(len(epochs)):
    #     for ii in range(64):
    #         data_signal = np.squeeze(epochs[i].get_data())
    #         freqs, psd = signal.welch(data_signal[ii], freq, nfft=win_size, nperseg=win_size, window='hamming')
    #
    #         freq_res = freqs[1] - freqs[0]
    #
    #         band_power = simps(psd[(freqs >= band[0]) & (freqs <= band[1])], dx=freq_res)
    #         # band_power = trapezoid(psd[(freqs >= band[0]) & (freqs <= band[1])], dx=freq_res)
    #
    #         band_power_epochs[i, ii] = band_power

    band_power_df = pd.DataFrame(data=band_power_epochs, index=event_recognized_df.index,
                                 columns=eeg_channel_band_power)
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
        event_df = event_df.join(bad_epochs) # creates nan if not processed at all
        epochs = epochs_ar

    if run_ica:
        # ICA parameters
        random_state = 42  # ensures ICA is reproducable each time it's run
        ica_n_components = 20  # Specify n_components as a decimal to set % explained variance

        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state, method='fastica',
                                    max_iter="auto").fit(epochs)
        ica.fit(epochs)
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
    event_df = event_df.join(band_power_df)

    return event_df, epochs, events, event_dict, info, reject_log, ica