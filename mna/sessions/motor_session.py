import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def process_session_motor(rns_data, event_df, motor_channel='Unity_MotorInput', save_path='../output/',
                          plot_motor_result=True, plot_motor_snippet=30, plot_frequency=10):
    if motor_channel in rns_data.keys():
        df = pd.DataFrame(rns_data[motor_channel][0], columns=rns_data[motor_channel][1],
                          index=rns_data[motor_channel][2]['ChannelNames']).T

        motor_start_time = df.index[0]
        motor_end_time = df.index[-1]
        count = 0
        motor_results = pd.DataFrame()
        for index, row in event_df.iterrows():
            if plot_frequency > 0 and count % plot_frequency == 0:
                plot_motor_result = True
            else:
                plot_motor_result = False
            timestamp_start = row['trial_start_time']
            timestamp_end = row['trial_end_time']

            timestamp_start = max(timestamp_start, motor_start_time)  # in case we don't have data starting from e.g. 0
            timestamp_end = min(timestamp_end, motor_end_time)  # in case the trial marker ends before eye data

            if plot_motor_result:
                plot_timestamp_end = min(timestamp_start + plot_motor_snippet, timestamp_end)  # plot 30 seconds of data
                plot_motor_data = df[(df.index >= timestamp_start) & (df.index <= plot_timestamp_end)]
                mot_fig = plot_motor_data.plot(
                    title=f"ppid_{row.ppid}_session_{row.session}_block_{row.block}_trial_{row.trial} Motor Results")
                plt.savefig(
                    f"{save_path}ppid_{row.ppid}_session_{row.session}_block_{row.block}_trial_{row.number_in_block}_motor.png")
                plt.close()

            motor_data = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]
            motor_results = pd.concat([motor_results, np.sum(np.abs(motor_data))], axis=1, ignore_index=True)
            count += 1
        motor_results = motor_results.T
        for mot_input_col in motor_results.columns:
            event_df[f"abs_sum_delta_{mot_input_col}"] = motor_results[mot_input_col]
        return event_df

    return event_df
