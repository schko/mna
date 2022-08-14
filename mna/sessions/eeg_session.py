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
<<<<<<< HEAD
                        hi_cut=30, plot_epochs=True):

    event_detected = event_df[event_column].notnull()
    event_recognized_df = event_df[event_detected]
=======
                        hi_cut=30, plot_epochs=True, bands_limits = [8,12]):
    voice_detected = event_df.spoken_difficulty.notnull()
    voice_recognized_df = event_df[voice_detected]
>>>>>>> 9a3587b (modified sessions)
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

<<<<<<< HEAD
    trial_start_time = event_recognized_df.trial_start_time - starting_time_s  # reference for mne
    event_values = event_recognized_df[event_column].values
=======
    trial_start_time = voice_recognized_df.trial_start_time - starting_time_s  # reference for mne
    spoken_difficulty = voice_recognized_df.spoken_difficulty.replace(to_replace=['easy', 'hard'],
                                                                      value=[1, 2])
>>>>>>> 9a3587b (modified sessions)

    events = np.column_stack((trial_start_time.values * freq,
                              np.zeros(len(event_recognized_df), dtype=int),
                              event_values)).astype(int)
    event_dict = dict(easy=1, hard=2)
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=- 0.2, tmax=3, preload=True)
    reject_log = None

    # Band power calculation
    win_size = 1024
    band = np.asarray(bands_limits)
    band_power = np.empty([len(epochs), 64])
    eeg_channel_names_alpha = [chan_name + "Band Power" for chan_name in eeg_channel_names]

    for i in range(len(epochs)):
        data_mne = np.squeeze(epochs[i].get_data())
        pow_freq_band = compute_pow_freq_bands(sfreq=freq, data=data_mne, freq_bands=band, normalize=False,
                                                    psd_params={'welch_n_fft': win_size, 'welch_n_per_seg': win_size})
        band_power[i, :] = pow_freq_band

    band_power_df = pd.DataFrame(data=band_power, index=voice_recognized_df.index, columns=eeg_channel_names_alpha)

    if run_autoreject:
        ar = autoreject.AutoReject(random_state=11,
                                   n_jobs=1, verbose=False)
        ar.fit(epochs[:autoreject_epochs])  # fit on a few epochs to save time
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        event_df.loc[event_detected, 'autorejected'] = reject_log.bad_epochs
        epochs = epochs_ar

    if run_ica:
        # ICA parameters
        random_state = 42  # ensures ICA is reproducable each time it's run
        ica_n_components = 20  # Specify n_components as a decimal to set % explained variance

        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state, method='fastica',
                                    max_iter="auto").fit(epochs)
        ica.fit(epochs)

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
        result_fig = reject_log.plot('horizontal')
        result_fig.savefig(
            f"{save_path}ppid_{ppid}_session_{session}_block_{block}_trial_{trial}_eeg_autoreject_preica.png")
        plt.close()
        ica.plot_components(show=False)
        plt.savefig(f"{save_path}ppid_{ppid}_session_{session}_block_{block}_trial_{trial}_eeg_ica.png")
        plt.close()

    event_df = event_df.join(band_power_df)

    return event_df, epochs, events, event_dict, info, reject_log, ica