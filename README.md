# Multimodal Neurophysiological Analysis

## MNA

A lightweight, Python package for analysis of eye tracking, electroencephalography (EEG) and electrocardiogram (ECG) data 
created for [LIINC](https://liinc.bme.columbia.edu/). Also supports epoch-level data quality reports and interactive review of data.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)

## Setup
Set up a conda environment by running the following command

```conda env create --name mna --file=environment.yml ```

Activate the environment

```conda activate mna```

## Usage

MNA requires two data structures: 
- a dictionary containing the raw data,
- an event matrix containing trial start and end timestamp, and 
- an event column that describes the trial or event.

Variable names below will refer to the example in `notebooks/process_session.ipynb`

### Raw Data

The primary data structure containing raw, time-series, data is a nested dictionary. The first level 

```python
[array([[ 5.18798828e-04,  5.18798828e-04,  5.18798828e-04, ...,
         -7.29370117e-03, -7.29370117e-03, -7.29370117e-03],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 3.05175781e-05,  3.05175781e-05,  3.05175781e-05, ...,
          3.05175781e-05,  3.05175781e-05,  3.05175781e-05],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
 array([2436.05086231, 2436.07311362, 2436.09533963, ..., 3767.04157113,
        3767.06377661, 3767.08597653]),
 {'StreamName': 'Unity.MotorInput',
  'ChannelNames': ['steer_input',
   'throttle_input',
   'brake_input',
   'clutch_input'],
  'NominalSamplingRate': 1}]
```
### Event DataFrame
