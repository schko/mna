import json
import pandas as pd

# pubsub from the rns data
# pd.DataFrame(rns_data['PubSub'][0], columns=rns_data['PubSub'][1],
#                  index=rns_data['PubSub'][2]['ChannelNames']).T
# merge with bigquery data

def read_bigquery(database_file):
    bigquery_df = pd.read_csv(database_file)
    subset_bigquery_df = bigquery_df[(bigquery_df.participant_id == 6) & (bigquery_df.session_no == 2)]
    datetime_cols = ['dataflow_publish_time', 'estimate_processing_start_timestamp', 'preprediction_timestamp',
                     'prediction_timestamp', 'window_start', 'window_end']
    subset_bigquery_df[datetime_cols] = subset_bigquery_df[datetime_cols].apply(pd.to_datetime)
    subset_bigquery_df = subset_bigquery_df.sort_values(by='window_start').reset_index(drop=True)
    subset_bigquery_df['lsl_timestamps'] = subset_bigquery_df.apply(
        lambda row: json.loads(row['trial_info'])[0]['timestamps'][0], axis=1)
    return subset_bigquery_df