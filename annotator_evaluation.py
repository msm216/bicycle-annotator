import os
import time
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from sklearn.metrics import confusion_matrix


# ********************************************** data loading ******************************************************** #

# find files of certain type under the current path
def find_file(kw="json"):
    path = os.getcwd()
    # list of all files
    content = os.listdir(path)
    f_list = []
    for file in content:
        if kw in file:
            f_list.append(file)

    return f_list


# list of json files unter the path
json_list = find_file("json")
print(
    "There are {0} json-files available: {1[0]} and {1[1]}".format(
        len(json_list), json_list
    )
)
# read anonymized_project.json
with open(json_list[0], "r") as f1:
    anno = json.load(f1)

# read references.json
with open(json_list[1], "r") as f2:
    ref = json.load(f2)
# dict, length 1
result = anno["results"]
# dict, length 2
root = result["root_node"]
# dict, length 9087
crowds = root["results"]

print("Number of crowds: {}".format(len(crowds)))
print("Number of references: {}".format(len(ref)))
print("=" * 50)


# ******************************************************************************************************************** #

# extract the needed information from each task
def crowd_extractor(crowd):
    """
    A crowd object must be a dictionary under the root_node['results'], which contains 10
    tasks/annotations as it's own 'result'. Tasks are contained in a LIST under the chosen
    crowd.
    For example:
    A crowd: crowds['7e8984b6-dff7-4015-865a-b721a2faf681'] contains 10 tasks (dictionary),
    which are contained in list: crowds['7e8984b6-dff7-4015-865a-b721a2faf681']['results'].
    The output of this function is a 'list of lists' with needed information about the
    tasks/annoations under the input crowd.
    """
    crowd_info = []

    for task in crowd["results"]:
        # extract information of the task
        url = task["task_input"]["image_url"]
        image = url.replace(
            "https://qm-auto-annotator.s3.eu-central-1.amazonaws.com/bicycles/", ""
        ).replace(".jpg", "")
        # name of annotators
        user = task["user"]["vendor_user_id"]
        # 'yes/no' to '1/0', NaN if no answer
        answer = task["task_output"]["answer"]
        # 'True/False' to '1/0'
        solvable = int(not task["task_output"]["cant_solve"])
        corrupt = int(task["task_output"]["corrupt_data"])
        # duration in millisecond
        duration = int(task["task_output"]["duration_ms"])
        # collect the information
        task_info = [image, user, answer, solvable, corrupt, duration]
        # add to upper list
        crowd_info.append(task_info)

    return crowd_info


# iterate over all crowds and build a DataFrame
def build_data(crowds=crowds, ref=ref):
    """
    The crowds object here means the root_node['results'], which contains 9087 dictionaries
    (crowd). Each crowd has its own individual key
    For example:
    A crowd: crowds['7e8984b6-dff7-4015-865a-b721a2faf681'] contains 10 tasks (dictionary),
    which are contained in list: crowds['7e8984b6-dff7-4015-865a-b721a2faf681']['results'].
    The output of this function is a 'pandas.DataFrame' with needed information from the
    hole data set.
    """
    info = []
    # set up timer
    start = time()
    total = len(crowds.keys())
    i = 0

    for key in crowds.keys():
        i += 1
        percent = round(1.0 * i / total * 100, 2)
        # extract crowd information
        crowd = crowds[str(key)]
        crowd_info = crowd_extractor(crowd)
        info.extend(crowd_info)
        print(
            "Processing crowd: {0}, {1} [{2}/{3}]".format(
                key, str(percent) + "%", i, total
            ),
            end="\r",
        )
        # time.sleep(0.01)
    print("Processing done! {:.2f} seconds passed.".format((time() - start)), end="\r")

    # build DataFrame
    result = pd.DataFrame(
        info, columns=["image", "user", "answer", "solvable", "corrupt", "duration"]
    )
    # quantify the answers
    num_encode = {"answer": {"yes": 1, "no": 0, "": np.nan}}
    result.replace(num_encode, inplace=True)
    # add column of references
    result["reference"] = result.apply(
        lambda x: int(ref[x["image"]]["is_bicycle"]), axis=1
    )
    # add column of correctness
    result["correct"] = result.apply(
        lambda x: int(x["answer"] == x["reference"]), axis=1
    )

    return result


# fix the non-positive durations
def fix_negativ(df, inst="user", targ="duration"):
    """
    Replace the non-positive value with the average of positive values
    of column 'duration' for each 'user'.
    """
    # make a copy of the DataFrame
    df_new = df.copy(deep=True)
    # deduplication by chosen column
    inst_list = list(set(df_new[inst]))

    for i in inst_list:
        df_sl = df_new.loc[df_new[inst] == str(i)]
        # index of non-positive values
        neg_ind = list(df_sl.loc[(df_sl[targ] <= 0)].index)
        # mean of normal values
        mean = df_sl[~(df_sl[targ] <= 0)][targ].mean()
        df_new.loc[neg_ind, targ] = mean
    print("Negative values fixed.\n")

    return df_new


# get statistical information by annotator or images/questions
def statistic_master(df, col="user"):
    """
    This function is used to get statistic information from the data set, like:
        - amount of annotations by selected category
        - amount of positive annotations
        - amount of correct annotations
        - accuracy of annotation by selected category
        - average duration
        - precision
        - recall
    Argument col could take 'user' or 'image'
    The output is a pandas.DataFrame.
    """
    info = []

    # deduplication of chosen column and sort
    inst_list = list(set(df[col]))
    inst_list.sort()
    # set up timer
    start = time()
    total = len(inst_list)
    t = 0
    print("Start processing statistic...It will take a while.")

    for inst in inst_list:
        t += 1
        p = round(1.0 * t / total * 100, 2)

        # collect the statistic informations
        amo_tot = len(df.loc[df[col] == inst])
        amo_pos = int(df["answer"].loc[df[col] == inst].sum())
        amo_corre = int(df["correct"].loc[df[col] == inst].sum())
        avg_dura = round(df["duration"].loc[df[col] == inst].mean(), 2)
        accu = round(df["correct"].loc[df[col] == inst].mean(), 2)
        # save in list
        inst_info = [inst, amo_tot, amo_pos, amo_corre, avg_dura, accu]
        # optional infomation
        if col == "user":
            y_anno = list(df["answer"].loc[df[col] == inst])
            y_true = list(df["reference"].loc[df[col] == inst])
            tn, fp, fn, tp = confusion_matrix(y_true, y_anno).ravel()
            prec = round(tp / (tp + fp), 2)
            rec = round(tp / (tp + fn), 2)
            # accu = round((tp+tn)/(tn+fp+fn+tp), 2)
            inst_info.extend([prec, rec])
        if col == "image":
            ref = df["reference"].loc[df[col] == inst].mean()
            inst_info.append(ref)

        info.append(inst_info)
        print(
            "Processing: {0}, {1} [{2}/{3}]".format(inst, str(p) + "%", t, total),
            end="\r",
        )
        # time.sleep(0.01)

    print("Processing done! {:.2f} seconds passed.\n".format((time() - start)))

    # save in DataFrame
    clms = []
    if col == "user":
        clms = [
            col,
            "total amount",
            "positive amount",
            "correct amount",
            "avg.duration",
            "accuracy",
            "precision",
            "recall",
        ]
    if col == "image":
        clms = [
            col,
            "total amount",
            "positive amount",
            "correct amount",
            "avg.duration",
            "accuracy",
            "reference",
        ]
    result = pd.DataFrame(info, columns=clms)
    """
    # add percent sign
    result.loc[:, 'accuracy'] = result.loc[:, 'accuracy'].apply(
        lambda x: "{:.2f}%".format(x)
    )
    """
    return result


# ***************************************** data cleaning ************************************************************ #

# build the DataFrame
data_full = build_data()
print("=" * 50)

# check out the data set
print("Basic information of the loaded data:")
print(data_full.info())
print("=" * 50)

# anootations without available answer
n_nan = len(data_full[data_full["answer"].isnull() == True])
# drop the rows without available answer (90849 left)
data = data_full.dropna(subset=["answer"])
print("There are {} annotation(s) without available answer.".format(n_nan))
print("=" * 50)

# check out the values
print("Value of column 'duration':")
print(data_full["duration"].describe())
print("=" * 50)

# find out negative duration
n_bad_dura = len(data_full.loc[data_full["duration"] <= 0])
print("There are {} annotation(s) with non-positive duration.".format(n_bad_dura))
print("=" * 50)

# fix the bad values
data_fix = fix_negativ(data)


# ***************************** How many annotators did contribute to the dataset? *********************************** #

# deduplication by 'user'
user_list = list(set(data_full["user"]))
user_list.sort()
print(
    "There are {} annotators in total did contribute to the dataset.".format(
        len(user_list)
    )
)
print("=" * 50)

# *************** Did all annotators produce the same amount of results, or are there differences? ******************* #

# get amount of results for each annotators
print("Amount of results contributed by each annotator:")
print(data_full.groupby(["user"], as_index=False).size())
print("=" * 50)

# visualize the amounts of results
fig1, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Amount of results produced by annotators", y=1.05, size=15)
sns.countplot(x="user", data=data_full.sort_values(by="user"))
ax.set_xlabel("Annotators", fontsize=10)
ax.set_ylabel("Amount of results", fontsize=10)
ax.xaxis.set_tick_params(rotation=90)
ax.grid(axis="y")
plt.show()


# ************************ Are there questions for which annotators highly disagree? ********************************* #

# get statistic for images (takes 6 min!!)
stats_image = statistic_master(data_fix, col="image")
print("=" * 50)

# amount of images, to those not a single annotator gave a positive answer
n_disagree = len(stats_image.loc[stats_image["positive amount"] == 0])
n_fn = len(
    stats_image.loc[
        (stats_image["positive amount"] == 0) & (stats_image["accuracy"] == 0)
    ]
)
print(
    "There are {} images/questions for which all annotators totally disagree.".format(
        n_disagree
    )
)
print(
    "{} of the images/questions are incorrectly as negative annotated (false negative).".format(
        n_fn
    )
)
print(
    [
        im
        for im in list(
            stats_image["image"].loc[
                (stats_image["positive amount"] == 0) & (stats_image["accuracy"] == 0)
            ]
        )
    ]
)
print("=" * 50)


# ************************* What are the average, min and max annotation times (durations)? ************************** #

print("To the entire data set:")
print("-" * 50)
print("Median duration {:.2f} [ms]".format(data_fix["duration"].median()))
print("Maximal duration {:.2f} [ms]".format(data_fix["duration"].max()))
print("Minimal duration {:.2f} [ms]".format(data_fix["duration"].min()))
print("Average duration {:.2f} [ms]".format(data_fix["duration"].mean()))
print("=" * 50)

# visualize the durations
fig2, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Durations by each annotators")
sns.regplot(
    x="user", y="duration", data=data_fix.sort_values(by="user"), fit_reg=False, ax=ax
)
ax.set_xlabel("Annotators", fontsize=10)
ax.set_ylabel("Duration [ms]", fontsize=10)
ax.xaxis.set_tick_params(rotation=90)
ax.grid(axis="y")
plt.show()


# *********************************** How often does each occur in the project *************************************** #

n = len(data_full)
# slice 'unsolvable' data
df_unsol = data_full.sort_values(by="user").loc[data_full["solvable"] == 0]
n_unsol = len(df_unsol)
# slice 'corrupted' data
df_corr = data_full.sort_values(by="user").loc[data_full["corrupt"] == 1]
n_corr = len(df_corr)

print("To the entire data set:")
print("-" * 50)
print(
    "{0} annotations were unsolvable ({1}%)".format(
        n_unsol, round((n_unsol / n) * 100, 4)
    )
)
print(
    "{0} annotations were corrupt ({1}%)".format(n_corr, round((n_corr / n) * 100, 4))
)
print("No trend detected.")
print("=" * 50)

colormap = plt.cm.RdBu
plt.figure(figsize=(10, 10))
plt.title("Pearson Correlation", y=1.05, size=15)
sns.heatmap(
    data_full[["answer", "solvable", "corrupt", "duration", "correct"]]
    .astype(float)
    .corr(),
    linewidths=1,
    vmax=1.0,
    vmin=-1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)
plt.show()


# ************************************ Is the reference set balanced? ************************************************ #

# about the balance of reference
n_true = list(data_full["reference"]).count(1)
true_rate = n_true / len(data_full)

# I'm not sure about this range
if true_rate >= 0.33 and true_rate <= 0.66:
    jug = "balanced"
else:
    jug = "unbalanced"
print(
    "There are {0} positive samples in the reference,which is {1}% of the hole data set.".format(
        n_true, round(true_rate * 100, 2)
    )
)
print("The reference is {0}.".format(jug))
print("=" * 50)

# visualize the amounts of results
fig3, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Amount of results produced by annotators", y=1.05, size=15)
sns.countplot(x="user", hue="reference", data=data_full.sort_values(by="user"))
ax.set_xlabel("Annotators", fontsize=10)
ax.set_ylabel("Amount of results", fontsize=10)
ax.xaxis.set_tick_params(rotation=90)
ax.grid(axis="y")
plt.show()


# ********************************* Can you identify good and bad annotators? **************************************** #

# get statistical information for users
stats_user = statistic_master(data_fix, col="user")
print("=" * 50)

# find out annotator with high accuracy and low avg.duration
accu_q1 = stats_user["accuracy"].quantile(q=0.25)
accu_q3 = stats_user["accuracy"].quantile(q=0.75)
dura_q1 = stats_user["avg.duration"].quantile(q=0.25)
dura_q3 = stats_user["avg.duration"].quantile(q=0.75)
top_users = stats_user.loc[
    (stats_user["accuracy"] >= accu_q3) & (stats_user["avg.duration"] <= dura_q1)
]
bad_users = stats_user.loc[
    (stats_user["accuracy"] <= accu_q1) & (stats_user["avg.duration"] >= dura_q3)
]

print(
    "Annotators, who had high accuracy and took low avg.annotation time are good.\n",
    list(top_users["user"]),
)
print(top_users)
print(
    "Annotators, who had low accuracy and took long avg.annotation time are bad.\n",
    list(bad_users["user"]),
)
print(bad_users)
print("=" * 50)

# visualize the accuracy and avg. annotation time by each annotator
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

# ax[0].set_title("Accuracy", y=1.05, size=15)
sns.barplot(x="user", y="accuracy", data=stats_user, palette="Blues_d", ax=ax[0])
ax[0].set_ylabel("Accuracy of annotation", fontsize=10)
ax[0].set_xlabel("Annotators", fontsize=10)
ax[0].xaxis.set_tick_params(rotation=90)
ax[0].set_ylim([0, 1])
ax[0].grid(axis="y")

# ax[1].set_title("Avg.duration", y=1.05, size=15)
sns.barplot(x="user", y="avg.duration", data=stats_user, palette="Oranges_d", ax=ax[1])
ax[1].set_ylabel("Avg.duration", fontsize=10)
ax[1].set_xlabel("Annotators", fontsize=10)
ax[1].xaxis.set_tick_params(rotation=90)
ax[1].set_ylim([1, 1750])
ax[1].grid(axis="y")

plt.show()
