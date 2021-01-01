import sys
import time
import pandas as pd
import numpy as np
import glob
import os
import csv
import gzip
from sklearn.preprocessing import MinMaxScaler

# "genoveva" not included because data is mixed up for some days - will fix after looking through notes
# subjects = ["yared"]
subjects = ["adonay", "yared", "yosias", "yuxiao"]

phone_data_dir = "data_files/phone_features/"
watch_data_dir = "data_files/watch_labels/"
actigraph_data_dir = "data_files/watch_actigraph_scores/"


def modify_columns_for_training(df):
    columns_to_drop = ["watch_acceleration:magnitude_stats:mean", "watch_acceleration:magnitude_stats:std",
                       "watch_acceleration:magnitude_stats:moment3", "watch_acceleration:magnitude_stats:moment4",
                       "watch_acceleration:magnitude_stats:percentile25",
                       "watch_acceleration:magnitude_stats:percentile50",
                       "watch_acceleration:magnitude_stats:percentile75",
                       "watch_acceleration:magnitude_stats:value_entropy",
                       "watch_acceleration:magnitude_stats:time_entropy",
                       "watch_acceleration:magnitude_spectrum:log_energy_band0",
                       "watch_acceleration:magnitude_spectrum:log_energy_band1",
                       "watch_acceleration:magnitude_spectrum:log_energy_band2",
                       "watch_acceleration:magnitude_spectrum:log_energy_band3",
                       "watch_acceleration:magnitude_spectrum:log_energy_band4",
                       "watch_acceleration:magnitude_spectrum:spectral_entropy",
                       "watch_acceleration:magnitude_autocorrelation:period",
                       "watch_acceleration:magnitude_autocorrelation:normalized_ac", "watch_acceleration:3d:mean_x",
                       "watch_acceleration:3d:mean_y", "watch_acceleration:3d:mean_z", "watch_acceleration:3d:std_x",
                       "watch_acceleration:3d:std_y", "watch_acceleration:3d:std_z", "watch_acceleration:3d:ro_xy",
                       "watch_acceleration:3d:ro_xz",
                       "watch_acceleration:3d:ro_yz", "watch_acceleration:spectrum:x_log_energy_band0",
                       "watch_acceleration:spectrum:x_log_energy_band1",
                       "watch_acceleration:spectrum:x_log_energy_band2",
                       "watch_acceleration:spectrum:x_log_energy_band3",
                       "watch_acceleration:spectrum:x_log_energy_band4",
                       "watch_acceleration:spectrum:y_log_energy_band0",
                       "watch_acceleration:spectrum:y_log_energy_band1",
                       "watch_acceleration:spectrum:y_log_energy_band2",
                       "watch_acceleration:spectrum:y_log_energy_band3",
                       "watch_acceleration:spectrum:y_log_energy_band4",
                       "watch_acceleration:spectrum:z_log_energy_band0",
                       "watch_acceleration:spectrum:z_log_energy_band1",
                       "watch_acceleration:spectrum:z_log_energy_band2",
                       "watch_acceleration:spectrum:z_log_energy_band3",
                       "watch_acceleration:spectrum:z_log_energy_band4",
                       "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0",
                       "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1",
                       "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2",
                       "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3",
                       "watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4",
                       "location:min_altitude", "location:max_altitude", "location:best_vertical_accuracy",
                       "lf_measurements:proximity", "lf_measurements:relative_humidity",
                       "lf_measurements:screen_brightness", "lf_measurements:temperature_ambient"]

    return df.drop(columns_to_drop, axis=1)


def generate_timestamp_for_phone_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1000000000)
    df.sort_values(by=['timestamp'], inplace=True)
    return df


def generate_timestamp_for_label_data(df):
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"],
                                     format="%m/%d/%Y %I:%M %p")
    cols = list(df.columns)
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df = df.drop(["Date", "Time"], axis=1)
    df.sort_values(by=['timestamp'], inplace=True)
    return df


def generate_timestamp_for_actigraph_data(df):
    df["inbed_timestamp"] = pd.to_datetime(df["In Bed Date"].astype(str) + " " + df["In Bed Time"],
                                           format="%m/%d/%Y %I:%M %p")
    df["outbed_timestamp"] = pd.to_datetime(df["Out Bed Date"].astype(str) + " " + df["Out Bed Time"],
                                            format="%m/%d/%Y %I:%M %p")
    df["onset_timestamp"] = pd.to_datetime(df["Onset Date"].astype(str) + " " + df["Onset Time"],
                                           format="%m/%d/%Y %I:%M %p")
    cols = list(df.columns)
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df = df.drop(
        ["In Bed Date", "Out Bed Date", "Onset Date", "In Bed Time", "Out Bed Time", "Onset Time", "Sleep Algorithm",
         "Latency"], axis=1)
    df.sort_values(by=['inbed_timestamp'], inplace=True)
    return df


def print_distances_between_timestamps(df):
    temp_timestamp = 0
    gap_count = 0
    for timestamp in df["timestamp"]:
        difference = timestamp - temp_timestamp
        if difference > 400:
            gap_count += 1
        temp_timestamp = timestamp
    print(str(gap_count) + "(s) gaps detected")


def print_data_statistics(df_p, df_w, df_a):
    total_ratio = 0
    for idx, row in df_a.iterrows():
        print("sleep period: " + str(idx))
        start_ts = row["inbed_timestamp"]
        end_ts = row["outbed_timestamp"]
        # print("start = " + str(start_ts) + ", end = " + str(end_ts))
        phone_count = 0
        watch_count = 0
        s_count = 0
        w_count = 0
        for timestamp in df_p["timestamp"]:
            if start_ts < timestamp < end_ts:
                phone_count += 1
        for timestamp, sleep_flag in zip(df_w["timestamp"], df_w["Sleep or Awake?"]):
            if start_ts < timestamp < end_ts:
                watch_count += 1
                if sleep_flag == "S":
                    s_count += 1
                elif sleep_flag == "W":
                    w_count += 1
        # print("phone data size = " + str(phone_count))
        # print("watch data size = " + str(watch_count))
        print("data ratio = " + str(phone_count * 100 / watch_count))
        total_ratio += phone_count * 100 / watch_count
        # print("S count = " + str(s_count))
        # print("W count = " + str(w_count))
    print("Average data ratio = " + str(total_ratio / (idx + 1)))


def timestamp_to_epoch(timestamp):
    return timestamp.values.astype(np.int64) // 1000000000


# Merges dataframes using the closes timestamp values. As of now the smaller data (phone data) is the main dataframe to
# to which some watch values are picked and merged too
def reindex_and_merge_by_timestamp(df_p, df_w):
    df_p.sort_values('timestamp', inplace=True)
    df_w.sort_values('timestamp', inplace=True)
    df_w1 = df_w.set_index('timestamp').reindex(df_p.set_index('timestamp').index, method='nearest').reset_index()
    merged_df = pd.merge(df_p, df_w1, on='timestamp')
    return merged_df


# # return value new_X_list is a list of dataframes which should be matched to each row
# # of the dataframe that is returned as the other value (df_a) for training
def generate_actigraphy_segments(df_p, df_a, subject):
    new_npX_list = []
    invalid_sleep_periods = []
    min_phone_observation_threshold = 70
    phone_observations = []
    for idx_a, row_a in df_a.iterrows():
        print("", end=f"\r{subject}: {idx_a / df_a.shape[0] * 100}%")
        start_ts = row_a["inbed_timestamp"]
        end_ts = row_a["outbed_timestamp"]
        # print("sleep period: " + str(idx_a))
        # print("start = " + str(start_ts) + ", end = " + str(end_ts))
        # print("Sleep duration = " + str(end_ts.timestamp() - start_ts.timestamp()))

        new_dfX = pd.DataFrame(columns=list(df_p.columns))
        for idx_p, row_p in df_p.iterrows():
            if start_ts < row_p["timestamp"] < end_ts:
                new_dfX = new_dfX.append(df_p.iloc[idx_p])
        new_dfX = new_dfX.reset_index(drop=True)
        phone_observations_in_curr_sp = new_dfX.shape[0]
        # print("Duration from phone = " + str(observations_from_phone * 60))
        if phone_observations_in_curr_sp < min_phone_observation_threshold:
            invalid_sleep_periods.append(idx_a)
        else:
            phone_observations.append(phone_observations_in_curr_sp)
            new_dfX['timestamp'] = timestamp_to_epoch(new_dfX['timestamp'])

            new_dfX = clean_and_normalize_data(new_dfX)
            # new_dfX.to_csv(
            #     'data_files/training_data/actigraph_model_training_data_X_' + subject + '_clean' + str(idx_a) + '.csv',
            #     index=False)
            new_npX_list.append(new_dfX.to_numpy())  # convert df to array before appending

    print("", end=f"\r{subject}: 100%")
    print("")
    # drop invalid sleep periods
    df_a.drop(invalid_sleep_periods, 0, inplace=True)
    print("sleep periods dropped: " + str(len(invalid_sleep_periods)))
    print_observations_statistic(phone_observations.copy())

    return new_npX_list, df_a, phone_observations


def print_observations_statistic(observations):
    print(observations)
    print("max observations: " + str(max(observations)))
    print("min observations(non zero periods): " + str(min(observations)))
    print("avg observations(non zero periods): " + str(sum(observations) / len(observations)))
    return max(observations)


def clean_and_normalize_data(df):
    # columns with all null values
    null_columns = df[df.columns[df.isnull().all()]].columns.values
    # fill with zero
    df[null_columns] = df[null_columns].fillna(0)

    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer(strategy='mean')  # for median imputation replace 'mean' with 'median'
    imp_mean.fit(df)
    df[df.columns] = imp_mean.transform(df[df.columns])

    # normalizing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


def load_data():
    # Dictionary where you access data based on subject names
    # Currently there is only one subject called "adonay"
    # so to access a dataframe with his data you search phone_data["adonay"]
    print("Loading dataset from CSV files...")
    phone_dic = {}
    watch_dic = {}
    actigraph_dic = {}
    for subject in subjects:
        df_phone = pd.read_csv(os.path.join(phone_data_dir, subject + "_features.csv.gz"), index_col=None, header=0)
        df_phone = modify_columns_for_training(df_phone)
        phone_dic[subject] = generate_timestamp_for_phone_data(df_phone)

        df_list = []
        for gz_file_name in glob.glob(os.path.join(watch_data_dir, subject + "_labels*.csv")):
            df_list.append(pd.read_csv(gz_file_name, index_col=None, header=2))
        df_watch = generate_timestamp_for_label_data(pd.concat(df_list, axis=0, ignore_index=True))
        df_watch = df_watch.rename({'Sleep or Awake?': 'sleep_or_wake'}, axis=1)
        df_watch = df_watch[['timestamp', 'sleep_or_wake']]
        watch_dic[subject] = df_watch

        df_list = []
        for gz_file_name in glob.glob(os.path.join(actigraph_data_dir, subject + "_actigraph*.csv")):
            df_list.append(pd.read_csv(gz_file_name, index_col=None, header=4))
        df_actigraph = generate_timestamp_for_actigraph_data(pd.concat(df_list, axis=0, ignore_index=True))
        df_actigraph.columns = [col.lower().replace(' ', '_') for col in df_actigraph.columns]
        actigraph_dic[subject] = df_actigraph
        df_list = []

    print("Finished loading dataset!")
    return phone_dic, watch_dic, actigraph_dic


def setup_training_data_for_sleep_model(dic_p, dic_w):
    df_list = []
    for subject in subjects:
        df_list.append(reindex_and_merge_by_timestamp(dic_p[subject], dic_w[subject]))

    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    # final_df.sort_values('timestamp', inplace=True)
    final_df['timestamp'] = timestamp_to_epoch(final_df['timestamp'])
    final_df = final_df.reset_index(drop=True)

    return final_df


def setup_training_data_for_actigraphy_model(dic_p, dic_a):
    print("generating actigraph training data. please wait...")
    print("Progress per subject:")
    npX_list = []
    dfy_list = []
    observations = []
    for subject in subjects:
        print(subject + ": ", end='')
        tempX, tempy, obs = generate_actigraphy_segments(dic_p[subject], dic_a[subject], subject)
        print("")
        npX_list.extend(tempX)
        dfy_list.append(tempy)
        observations.extend(obs)

    window_max = max(observations)
    window = 150
    # resize arrays to fit window size
    complete_npX = []
    for arr in npX_list:
        window_fit_arr = fit_window(arr, window)
        complete_npX.append(window_fit_arr)

    print(len(complete_npX))

    complete_npX = np.dstack(complete_npX)  # stack arrays into a 3d array
    complete_npX = complete_npX.transpose(2, 0, 1)  # transpose array to fit dimensions needed by LSTM layers

    complete_dfy = pd.concat(dfy_list, axis=0, ignore_index=True)
    complete_dfy = complete_dfy.drop(["onset_timestamp", "inbed_timestamp", "outbed_timestamp"], axis=1)

    # TEST
    complete_dfy = complete_dfy["wake_after_sleep_onset_(waso)"]

    complete_dfy = complete_dfy.reset_index(drop=True)
    # complete_dfy.to_csv('data_files/training_data/actigraph_model_training_data_y.csv', index=False)

    print("finished generating actigraph training data")

    return complete_npX, complete_dfy.to_numpy()


def fit_window(arr, window):
    if arr.shape[0] >= window:
        # decrease rows in array its bigger than window size
        window_fit_arr = arr.copy()
        window_fit_arr.resize((window, arr.shape[1]))
    else:
        # pad array with -1 to fit window dimension
        window_fit_arr = np.pad(arr, [(0, window - arr.shape[0]), (0, 0)], mode='constant', constant_values=-1)

    return window_fit_arr


def check_sleep_label_balance(df):
    print(df.groupby('sleep_or_wake').count())


def get_sleep_model_training_data():
    phone_data, watch_data, actigraph_data = load_data()
    return setup_training_data_for_sleep_model(phone_data, watch_data)


def get_actigraphy_model_training_data():
    phone_data, watch_data, actigraph_data = load_data()
    return setup_training_data_for_actigraphy_model(phone_data, actigraph_data)


def test():
    # df = get_sleep_model_training_data()
    # check_sleep_label_balance(df)
    test_subject = "yosias"
    phone_data, watch_data, actigraph_data = load_data()
    X, y, obs = generate_actigraphy_segments(phone_data[test_subject], actigraph_data[test_subject],
                                             test_subject)

    # print_data_statistics(phone_data[test_subject], watch_data[test_subject], actigraph_data[test_subject])


if __name__ == '__main__':
    test()
