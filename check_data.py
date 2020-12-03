import pandas as pd
import numpy as np
import glob
import os
import csv
import gzip

# subjects = ["adonay", "yared", "yosias", "yuxiao", "", ""]
subjects = ["adonay"]

phone_data_dir = "data_files/phone_features/"
watch_data_dir = "data_files/watch_labels/"
actigraph_data_dir = "data_files/watch_actigraph_scores/"

print("Reading wash_original csv files")


def generate_timestamp_for_label_data(df):
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"],
                                     format="%m/%d/%Y %I:%M %p").values.astype(np.int64) // 1000000000
    cols = list(df.columns)
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df = df.drop(["Date", "Time"], axis=1)
    df.sort_values(by=['timestamp'], inplace=True)
    return df


def generate_timestamp_for_actigraph_data(df):
    df["inbed_timestamp"] = pd.to_datetime(df["In Bed Date"].astype(str) + " " + df["In Bed Time"],
                                           format="%m/%d/%Y %I:%M %p").values.astype(np.int64) // 1000000000
    df["outbed_timestamp"] = pd.to_datetime(df["Out Bed Date"].astype(str) + " " + df["Out Bed Time"],
                                            format="%m/%d/%Y %I:%M %p").values.astype(np.int64) // 1000000000
    df["onset_timestamp"] = pd.to_datetime(df["Onset Date"].astype(str) + " " + df["Onset Time"],
                                           format="%m/%d/%Y %I:%M %p").values.astype(np.int64) // 1000000000
    cols = list(df.columns)
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df = df.drop(["In Bed Date", "Out Bed Date", "Onset Date", "In Bed Time", "Out Bed Time", "Onset Time"], axis=1)
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


# def separate_watch_data_into_sleep_periods(df_w):


def merge_phone_and_watch_data(df_p, df_w, df_a):
    for idx, row in df_a.iterrows():
        print("sleep period: " + str(idx))
        start_ts = row["inbed_timestamp"]
        end_ts = row["outbed_timestamp"]
        print("start = " + str(start_ts) + ", end = " + str(end_ts))
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
        print("data ratio = " + str(phone_count*100/watch_count))
        # print("S count = " + str(s_count))
        # print("W count = " + str(w_count))


def load_data():
    phone_dic = {}
    watch_dic = {}
    actigraph_dic = {}
    for subject in subjects:
        df_phone = pd.read_csv(os.path.join(phone_data_dir, subject + "_features.csv.gz"), index_col=None, header=0)
        df_phone.sort_values(by=['timestamp'], inplace=True)
        phone_dic[subject] = df_phone
        df_watch = generate_timestamp_for_label_data(
            pd.read_csv(os.path.join(watch_data_dir, subject + "_labels.csv"), index_col=None, header=2))
        watch_dic[subject] = df_watch
        df_actigraph = generate_timestamp_for_actigraph_data(
            pd.read_csv(os.path.join(actigraph_data_dir, subject + "_actigraph.csv"), index_col=None,
                        header=4))
        actigraph_dic[subject] = df_actigraph

    return phone_dic, watch_dic, actigraph_dic


if __name__ == '__main__':
    phone_data, watch_data, actigraph_data = load_data()
    merge_phone_and_watch_data(phone_data["adonay"], watch_data["adonay"], actigraph_data["adonay"])
    print_distances_between_timestamps(phone_data["adonay"])
