import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# from: https://gist.github.com/sixtenbe/1178136
def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
    
    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    
    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx, y_axis[index:index+lookahead]])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn, y_axis[index:index+lookahead]])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]

def process_session_motor(rns_data, event_df, motor_channel='Unity_MotorInput', save_path='../output/',
                          plot_motor_result=True, plot_motor_snippet=30, plot_frequency=10, freq=40, turn_lookahead=750, preturn=1000, postturn=0):
    # preturn and postturn period in mseconds
    if motor_channel in rns_data.keys():
        df = pd.DataFrame(rns_data[motor_channel][0], columns=rns_data[motor_channel][1],
                          index=rns_data[motor_channel][2]['ChannelNames']).T

        motor_start_time = df.index[0]
        motor_end_time = df.index[-1]
        count = 0
        motor_results = pd.DataFrame()
        turns_df = []
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
    
                # wheel is 40Hz, so turn_lookahead/freq 750 ms of lookahead and 18 degrees of rotation (10%) empirically does okay
                _max, _min = peakdetect(plot_motor_data.steer_input, plot_motor_data.index, lookahead=int((turn_lookahead/1000)*freq),
                                        delta=0.1)
                
                mot_fig = plot_motor_data.plot(
                    title=f"ppid_{row.ppid}_session_{row.session}_block_{row.block}_trial_{row.trial} Motor Results")
                plt.scatter(x=[ts[0] for ts in _max], y=[ts[1] for ts in _max], color='r')
                plt.scatter(x=[ts[0] for ts in _min], y=[ts[1] for ts in _min], color='r')
                plt.savefig(
                    f"{save_path}ppid_{row.ppid}_session_{row.session}_block_{row.block}_trial_{row.number_in_block}_motor.png")
                plt.close()

            motor_data = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]
            # wheel is 40Hz, so turn_lookahead==750ms and 18 degrees of rotation (10%) empirically does okay
            _max, _min = peakdetect(motor_data.steer_input, motor_data.index, lookahead=int((turn_lookahead/1000)*freq),
                                    delta=0.1)
            
            for peak in _max:
                turn_row = row.to_dict().copy()
                if turn_row['trial_start_time'] <= peak[0]-preturn/1000 and turn_row['trial_end_time'] >= peak[0]+postturn/1000: # ensure there's enough data
                    turn_row['trial_start_time'] = peak[0]-preturn/1000
                    turn_row['trial_end_time'] = peak[0] # note that this is a misnomer now
                    turn_row['post_steer_event_raw'] = peak[2]
                    turn_row['turn_type'] = 'left'
                    turns_df.append(turn_row)
            for trough in _min:
                turn_row = row.to_dict().copy()
                if turn_row['trial_start_time'] <= trough[0]-preturn/1000 and turn_row['trial_end_time'] >= trough[0]+postturn/1000:
                    turn_row['trial_start_time'] = trough[0]-preturn/1000
                    turn_row['trial_end_time'] = trough[0]
                    turn_row['post_steer_event_raw'] = trough[2]
                    turn_row['turn_type'] = 'right'
                    turns_df.append(turn_row)
            motor_results = pd.concat([motor_results, np.sum(np.abs(motor_data))], axis=1, ignore_index=True)
            count += 1
        motor_results = motor_results.T
        turns_df = pd.DataFrame(turns_df)
        for mot_input_col in motor_results.columns:
            event_df[f"abs_sum_delta_{mot_input_col}"] = motor_results[mot_input_col]
            turns_df[f"abs_sum_delta_{mot_input_col}"] = motor_results[mot_input_col]
        return event_df, turns_df

    return event_df, turns_df