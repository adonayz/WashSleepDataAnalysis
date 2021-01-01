import pandas as pd

dataset = pd.read_csv("data_files/training_data/sleep_model_training_data.csv", index_col=None, header=0)
dataset.sort_values(by=['timestamp'], inplace=True)

# pred:Sleep or predXGB:Sleep
sleep_column_used = 'sleep_or_wake'

sleep_threshold = 4  # in minutes
awake_threshold = 160  # in seconds
sleep_period_threshold = 10  # in minutes

new_period_flag = 0
new_period_count = 0

# sleep features
total_sleep_time = 0
nwaks = 0
total_awake_time = 0
avg_awakening = 0
time_in_bed = 0

# variables
consecutive_awake_count_in_sleep = 0
isSleeping = 0
start_sleep_time = 0
end_sleep_time = 0
start_awake_time = 0
end_awake_time = 0

sleep_quality_features = []

df_pred = dataset[sleep_column_used]


def reset_variables():
    # sleep features
    global total_sleep_time
    global nwaks
    global total_awake_time
    global avg_awakening
    global time_in_bed

    # variables
    global consecutive_awake_count_in_sleep
    global isSleeping
    global start_sleep_time
    global end_sleep_time
    global start_awake_time
    global end_awake_time

    # sleep features
    total_sleep_time = 0
    nwaks = 0
    total_awake_time = 0
    avg_awakening = 0
    time_in_bed = 0

    # variables
    consecutive_awake_count_in_sleep = 0
    isSleeping = 0
    start_sleep_time = 0
    end_sleep_time = 0
    start_awake_time = 0
    end_awake_time = 0


for index, row in dataset.iterrows():

    ## method1
    ## USING PREDICTED SLEEP VALUES TO SEPARATE SLEEP PERIODS
    #
    #
    #
    if row[sleep_column_used] == 0 and new_period_flag == 1:
        isEnd = 1
        for x in range(awake_threshold):
            if df_pred.iloc[index + x]:
                isEnd = 0
                break

        if isEnd:
            new_period_flag = 0
            new_features = {}
            new_features["total_sleep_time"] = total_sleep_time / 60
            new_features["nwaks"] = nwaks
            new_features["total_awake_time"] = total_awake_time / 60
            new_features["average_awakenings"] = total_awake_time / (60 * nwaks)
            new_features["time_in_bed"] = (total_sleep_time + total_awake_time) / 60
            new_features["sleep_efficiency"] = (total_sleep_time / (total_sleep_time + total_awake_time)) * 100
            # new_features["sol"] = (total_sleep_time + total_awake_time) / 60
            # new_features["twak"] = (total_sleep_time / (total_sleep_time + total_awake_time)) * 100
            sleep_quality_features.append(new_features)

            reset_variables()

    if row[sleep_column_used] == 1 and new_period_flag == 0:
        isStart = 1
        for j in range(sleep_period_threshold):
            if df_pred.iloc[index + j] == 0:
                isStart = 0
                break

        if isStart:
            new_period_count += 1
            new_period_flag = 1
    # end of method 1

    # ## method2
    # ## USING ACTIGRAPH SLEEP DURATION TO SEPARATE SLEEP PERIODS
    # #
    # #
    # #
    # if row['SleepPeriod'] == 1:
    #     if new_period_flag == 0:
    #         new_period_count += 1
    #         new_period_flag = 1
    # else:
    #     if new_period_flag == 1:
    #         new_features = {}
    #         new_features["total_sleep_time"] = total_sleep_time
    #         new_features["total_awake_time"] = total_awake_time
    #         new_features["time_in_bed"] = total_awake_time + total_sleep_time
    #         new_features["nwaks"] = nwaks
    #         new_features["average_awakenings"] = total_awake_time / nwaks
    #         new_features["time_in_bed"] = (total_sleep_time + total_awake_time)/60
    #         new_features["sleep_efficiency"] = (total_sleep_time/(total_sleep_time + total_awake_time))*100
    #         sleep_quality_features.append(new_features)
    #
    #         reset_variables()
    #     new_period_flag = 0
    # ## End of method 2

    if new_period_flag == 1:

        if row[sleep_column_used] == 1:
            consecutive_awake_count_in_sleep = 0
        else:
            consecutive_awake_count_in_sleep += 1

        if consecutive_awake_count_in_sleep >= sleep_threshold:
            # subject is awake
            if isSleeping == 1:
                # subject just woke up
                end_sleep_time = row['timestamp']
                total_sleep_time += (end_sleep_time - start_sleep_time)
                nwaks += 1
                start_awake_time = row['timestamp']
            # else:
            #     # subject has been awake

            isSleeping = 0
        else:
            # subject is asleep
            if isSleeping == 0:
                # subject just started sleeping
                start_sleep_time = row['timestamp']
                if start_awake_time != 0:
                    total_awake_time += (start_sleep_time - start_awake_time)
                    # end_awake_time = row['timestamp']
                    # total_awake_time += end_awake_time - start_awake_time
            # else:
            #     # subject has been sleeping

            isSleeping = 1

print ("Sleep periods = " + str(new_period_count))
print ("Sleep quality report")
for a_period in sleep_quality_features:
    print(a_period)
