import glob
import collections
import datetime
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# data_folder = 'baseline_results/'
data_folder = 'baseline_results_per_annotator/'
color_set = ['y','r','b','g','b']
color_index = 0
# skip train2 and train 5 for better plotting
trainset_labels = ['set-A', 'set-B', 'set-C']

# test_data = pd.read_csv("unreleased_ia_filtered_with_score_reversed_filtered.csv")
# test_data = test_data.dropna(axis=0, subset = ['Overall'])
# y_test = test_data['Overall']
test_data = pd.read_csv("per_annotator_evaluation_data.csv")
test_data = test_data.dropna(axis=0, subset = ['OVERALL'])
y_test = test_data['OVERALL']

files=list(glob.glob(data_folder + '*.csv'))
plot_data = collections.defaultdict(dict)
modified_data = collections.defaultdict(list)

for file in files:
    print(file, datetime.datetime.now(), flush=True)
    extract_content = file.split('/')[-1].split('.')[0].split('baseline_prediction_')[-1]

    train_set = extract_content.split("_")[0][-1]
    model_name = extract_content.split(train_set + '_')[-1]

    cur_predict = pd.read_csv(file)
    plot_data[model_name][train_set] = pearsonr(y_test, cur_predict['Overall'])[0]

# plot
model_keys = list(plot_data.keys())
trainset_keys = list(plot_data[model_keys[0]].keys())
x = list(range(len(trainset_keys)-2))

for model_name in model_keys:
    for i in range(1, len(plot_data[model_name]) + 1):
        if i == 2 or i == 5:
            continue
        modified_data[model_name].append(plot_data[model_name][str(i)])

for model_name in model_keys:
    total_width, n = 0.8, 4
    width = total_width/n
    a = plot_data[model_name].values()

    if model_name == 'xgboost':
        plt.bar(x, modified_data[model_name], width=width, label = model_name, tick_label = trainset_labels, fc=color_set[color_index])
    else:
        plt.bar(x, modified_data[model_name], width=width, label=model_name, fc=color_set[color_index])
    color_index += 1
    for i in range(len(x)):
        x[i] += width

plt.xticks(fontsize=16)
plt.ylabel('pearson correlation', fontsize=16)
plt.legend(fontsize=16)

# plt.title("model performance with different features (data as per pair)")
# plt.title("model performance with different features (data as per annotation)")
# plt.savefig(data_folder + 'baseline_summary.png')
plt.savefig(data_folder + 'baseline_summary_per_annotator.png')





# color_set = ['y','r','b','g','b']
# color_index = 0
# # skip train2 for better plotting
# trainset_labels = ['features_A', 'features_B', 'features_C', 'features_D']
#
# # data_folder = 'baseline_results/'
# # # data_folder = 'baseline_results_per_annotator/'
# #
# # test_data = pd.read_csv("unreleased_ia_filtered_with_score_reversed_filtered.csv")
# # test_data = test_data.dropna(axis=0, subset = ['Overall'])
# # y_test = test_data['Overall']
# # # test_data = pd.read_csv("per_annotator_evaluation_data.csv")
# # # test_data = test_data.dropna(axis=0, subset = ['OVERALL'])
# # # y_test = test_data['OVERALL']
#
# # files=list(glob.glob(data_folder + '*.csv'))
# # plot_data = collections.defaultdict(dict)
# # modified_data = collections.defaultdict(list)
# #
# # for file in files:
# #     print(file, datetime.datetime.now(), flush=True)
# #     extract_content = file.split('/')[-1].split('.')[0].split('baseline_prediction_')[-1]
# #
# #     train_set = extract_content.split("_")[0][-1]
# #     model_name = extract_content.split(train_set + '_')[-1]
# #
# #     cur_predict = pd.read_csv(file)
# #     plot_data[model_name][train_set] = pearsonr(y_test, cur_predict['Overall'])[0]
#
# dataset_name = ['per_item','per_annotator',"train_item_test_annotator", "train_annotator_test_item"]
# data_folders = ['baseline_results_per_item/', 'baseline_results_per_annotator/','baseline_results_train_per_item_test_per_annotator/','baseline_results_train_per_annotator_test_per_item/']
# test_datas = ["unreleased_ia_filtered_with_score_reversed_filtered.csv", "per_annotator_evaluation_data.csv", "per_annotator_evaluation_data.csv", "unreleased_ia_filtered_with_score_reversed_filtered.csv"]
# Overall_attribute_type = ['Overall', 'OVERALL', 'OVERALL', 'Overall']
#
# plot_data = []
# modified_data = []
#
# for cur_i in range(len(data_folders)):
#     data_folder = data_folders[cur_i]
#     test_data = pd.read_csv(test_datas[cur_i])
#     test_data = test_data.dropna(axis=0, subset=[Overall_attribute_type[cur_i]])
#     y_test = test_data[Overall_attribute_type[cur_i]]
#
#
#     files=list(glob.glob(data_folder + '*.csv'))
#     cur_plot_data = collections.defaultdict(dict)
#
#     for file in files:
#         print(file, datetime.datetime.now(), flush=True)
#         extract_content = file.split('/')[-1].split('.')[0].split('baseline_prediction_')[-1]
#
#         train_set = extract_content.split("_")[0][-1]
#         model_name = extract_content.split(train_set + '_')[-1]
#
#         cur_predict = pd.read_csv(file)
#         cur_plot_data[model_name][train_set] = pearsonr(y_test, cur_predict['Overall'])[0]
#     plot_data.append(cur_plot_data)
#     modified_data.append(collections.defaultdict(list))
#
#
#
# # plot
# model_keys = list(plot_data[0].keys())
# trainset_keys = list(plot_data[0][model_keys[0]].keys())
# x = list(range(len(trainset_keys)-1))
#
# for cur_i in range(len(data_folders)):
#     for model_name in model_keys:
#         for i in range(1, len(plot_data[cur_i][model_name]) + 1):
#             if i == 2:
#                 continue
#             modified_data[cur_i][model_name].append(plot_data[cur_i][model_name][str(i)])
#
#
# for cur_i in range(len(data_folders)):
#     for model_name in model_keys:
#         total_width, n = 0.8, 4
#         width = total_width/n
#         # a = plot_data[model_name].values()
#
#         plt.bar(x, modified_data[cur_i][model_name], width=width, label = dataset_name[cur_i], tick_label = trainset_labels, fc=color_set[color_index])
#         color_index += 1
#         for i in range(len(x)):
#             x[i] += width
#
# plt.ylabel('pearson correlation')
# plt.legend()
#
# # plt.title("baseline performance with different dataset and features")
# plt.savefig('baseline_summary.png')
#
# # plt.title("baseline performance with different features (data as per pair)")
# # # plt.title("model performance with different features (data as per annotation)")
# # plt.savefig(data_folder + 'baseline_summary.png')
# # # plt.savefig(data_folder + 'baseline_summary_per_annotator.png')


