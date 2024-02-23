# this is to integrate the three datasets in mediacloud dataset into MBFC dataset, we regard all the publishers with bias in mediacloud are also in US (most of them actually are)

import pandas as pd
from collections import Counter
from utils import unify_url, bias_class, valid_bias_class
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



# for test duplicates
a = pd.read_csv("/Users/xichen/Documents/GitHub/mediacloud/ner_art_sampling/integrated_bias.csv")
b = a['link'].to_list()
c = Counter(b)



raw_mbfc = pd.read_csv("mbfc_sources.csv")


# 2016 election
election_bias_folder = "bias_dataset/2016_election_bias/"

cur_center = pd.read_csv(election_bias_folder + "Tweeted_Evenly_by_Trump_Clinton_Followers_2016__US_Center_2016.csv")
cur_left = pd.read_csv(election_bias_folder + "Tweeted_Mostly_by_Clinton_Followers_2016__US_Left_2016.csv")
cur_right = pd.read_csv(election_bias_folder + "Tweeted_Mostly_by_Trump_Followers_2016__US_Right_2016.csv")
cur_left_center = pd.read_csv(election_bias_folder + "Tweeted_Somewhat_More_by_Clinton_Followers_2016__US_Center_Left_2016.csv")
cur_right_center = pd.read_csv(election_bias_folder + "Tweeted_Somewhat_More_by_Trump_Followers_2016__US_Center_Right_2016.csv")

cur_center.rename(columns={'url':'link'}, inplace=True)
cur_left.rename(columns={'url':'link'}, inplace=True)
cur_right.rename(columns={'url':'link'}, inplace=True)
cur_left_center.rename(columns={'url':'link'}, inplace=True)
cur_right_center.rename(columns={'url':'link'}, inplace=True)

cur_center.rename(columns={'pub_country':'country'}, inplace=True)
cur_left.rename(columns={'pub_country':'country'}, inplace=True)
cur_right.rename(columns={'pub_country':'country'}, inplace=True)
cur_left_center.rename(columns={'pub_country':'country'}, inplace=True)
cur_right_center.rename(columns={'pub_country':'country'}, inplace=True)

cur_center['bias'] = cur_center.apply(lambda x: 'center', axis=1)
cur_left['bias'] = cur_left.apply(lambda x: 'left', axis=1)
cur_right['bias'] = cur_right.apply(lambda x: 'right', axis=1)
cur_left_center['bias'] = cur_left_center.apply(lambda x: 'left_center', axis=1)
cur_right_center['bias'] = cur_right_center.apply(lambda x: 'right_center', axis=1)

cur_center = cur_center.loc[:,['link','name','country','media_type','bias']]
cur_left = cur_left.loc[:,['link','name','country','media_type','bias']]
cur_right = cur_right.loc[:,['link','name','country','media_type','bias']]
cur_left_center = cur_left_center.loc[:,['link','name','country','media_type','bias']]
cur_right_center = cur_right_center.loc[:,['link','name','country','media_type','bias']]

raw_mbfc_link_list = raw_mbfc['link'].to_list()
cur_center = cur_center[~cur_center['link'].isin(raw_mbfc_link_list)]
cur_left = cur_left[~cur_left['link'].isin(raw_mbfc_link_list)]
cur_right = cur_right[~cur_right['link'].isin(raw_mbfc_link_list)]
cur_left_center = cur_left_center[~cur_left_center['link'].isin(raw_mbfc_link_list)]
cur_right_center = cur_right_center[~cur_right_center['link'].isin(raw_mbfc_link_list)]

raw_mbfc = raw_mbfc.append(cur_center, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_left, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_right, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_left_center, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_right_center, ignore_index=True)

# 2018 vote bias
vote_bias_folder = "bias_dataset/2018_vote_bias/"

cur_center = pd.read_csv(vote_bias_folder + "Tweeted_Evenly_by_Republican_Democrat_Voters_2018.csv")
cur_left = pd.read_csv(vote_bias_folder + "Tweeted_Mostly_by_Democrat_Voters_2018.csv")
cur_right = pd.read_csv(vote_bias_folder + "Tweeted_Mostly_by_Republican_Voters_2018.csv")
cur_left_center = pd.read_csv(vote_bias_folder + "Tweeted_Somewhat_More_by_Democrat_Voters_2018.csv")
cur_right_center = pd.read_csv(vote_bias_folder + "Tweeted_Somewhat_More_by_Republican_Voters_2018.csv")

cur_center.rename(columns={'url':'link'}, inplace=True)
cur_left.rename(columns={'url':'link'}, inplace=True)
cur_right.rename(columns={'url':'link'}, inplace=True)
cur_left_center.rename(columns={'url':'link'}, inplace=True)
cur_right_center.rename(columns={'url':'link'}, inplace=True)

cur_center.rename(columns={'pub_country':'country'}, inplace=True)
cur_left.rename(columns={'pub_country':'country'}, inplace=True)
cur_right.rename(columns={'pub_country':'country'}, inplace=True)
cur_left_center.rename(columns={'pub_country':'country'}, inplace=True)
cur_right_center.rename(columns={'pub_country':'country'}, inplace=True)

cur_center['bias'] = cur_center.apply(lambda x: 'center', axis=1)
cur_left['bias'] = cur_left.apply(lambda x: 'left', axis=1)
cur_right['bias'] = cur_right.apply(lambda x: 'right', axis=1)
cur_left_center['bias'] = cur_left_center.apply(lambda x: 'left_center', axis=1)
cur_right_center['bias'] = cur_right_center.apply(lambda x: 'right_center', axis=1)

cur_center = cur_center.loc[:,['link','name','country','media_type','bias']]
cur_left = cur_left.loc[:,['link','name','country','media_type','bias']]
cur_right = cur_right.loc[:,['link','name','country','media_type','bias']]
cur_left_center = cur_left_center.loc[:,['link','name','country','media_type','bias']]
cur_right_center = cur_right_center.loc[:,['link','name','country','media_type','bias']]

raw_mbfc_link_list = raw_mbfc['link'].to_list()
cur_center = cur_center[~cur_center['link'].isin(raw_mbfc_link_list)]
cur_left = cur_left[~cur_left['link'].isin(raw_mbfc_link_list)]
cur_right = cur_right[~cur_right['link'].isin(raw_mbfc_link_list)]
cur_left_center = cur_left_center[~cur_left_center['link'].isin(raw_mbfc_link_list)]
cur_right_center = cur_right_center[~cur_right_center['link'].isin(raw_mbfc_link_list)]

raw_mbfc = raw_mbfc.append(cur_center, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_left, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_right, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_left_center, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_right_center, ignore_index=True)


# 2019 politician bias
politician_bias_folder = "bias_dataset/2019_politician_bias/"

cur_center = pd.read_csv(politician_bias_folder + "Tweeted_Evenly_by_Followers_of_Conservative___Liberal_Politicians_2019__US_Center_2019.csv")
cur_left = pd.read_csv(politician_bias_folder + "Tweeted_Mostly_by_Followers_of_Liberal_Politicians_2019__US_Left_2019.csv")
cur_right = pd.read_csv(politician_bias_folder + "Tweeted_Mostly_by_Followers_of_Conservative_Politicians_2019__US_Right_2019.csv")
cur_left_center = pd.read_csv(politician_bias_folder + "Tweeted_Somewhat_More_by_Followers_of_Liberal_Politicians_2019__US_Center_Left_2019.csv")
cur_right_center = pd.read_csv(politician_bias_folder + "Tweeted_Somewhat_More_by_Followers_of_Conservative_Politicians_2019__US_Center_Right_2019.csv")

cur_center.rename(columns={'url':'link'}, inplace=True)
cur_left.rename(columns={'url':'link'}, inplace=True)
cur_right.rename(columns={'url':'link'}, inplace=True)
cur_left_center.rename(columns={'url':'link'}, inplace=True)
cur_right_center.rename(columns={'url':'link'}, inplace=True)

cur_center.rename(columns={'pub_country':'country'}, inplace=True)
cur_left.rename(columns={'pub_country':'country'}, inplace=True)
cur_right.rename(columns={'pub_country':'country'}, inplace=True)
cur_left_center.rename(columns={'pub_country':'country'}, inplace=True)
cur_right_center.rename(columns={'pub_country':'country'}, inplace=True)

cur_center['bias'] = cur_center.apply(lambda x: 'center', axis=1)
cur_left['bias'] = cur_left.apply(lambda x: 'left', axis=1)
cur_right['bias'] = cur_right.apply(lambda x: 'right', axis=1)
cur_left_center['bias'] = cur_left_center.apply(lambda x: 'left_center', axis=1)
cur_right_center['bias'] = cur_right_center.apply(lambda x: 'right_center', axis=1)

cur_center = cur_center.loc[:,['link','name','country','media_type','bias']]
cur_left = cur_left.loc[:,['link','name','country','media_type','bias']]
cur_right = cur_right.loc[:,['link','name','country','media_type','bias']]
cur_left_center = cur_left_center.loc[:,['link','name','country','media_type','bias']]
cur_right_center = cur_right_center.loc[:,['link','name','country','media_type','bias']]

raw_mbfc_link_list = raw_mbfc['link'].to_list()
cur_center = cur_center[~cur_center['link'].isin(raw_mbfc_link_list)]
cur_left = cur_left[~cur_left['link'].isin(raw_mbfc_link_list)]
cur_right = cur_right[~cur_right['link'].isin(raw_mbfc_link_list)]
cur_left_center = cur_left_center[~cur_left_center['link'].isin(raw_mbfc_link_list)]
cur_right_center = cur_right_center[~cur_right_center['link'].isin(raw_mbfc_link_list)]

raw_mbfc = raw_mbfc.append(cur_center, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_left, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_right, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_left_center, ignore_index=True)
raw_mbfc = raw_mbfc.append(cur_right_center, ignore_index=True)

raw_mbfc['link'] = raw_mbfc.apply(lambda x: unify_url(x['link']), axis=1)
raw_mbfc['bias_index'] = raw_mbfc.apply(lambda x: bias_class.index(x['bias']), axis=1)
raw_mbfc['grouped_bias_index'] = raw_mbfc.groupby(['link'])['bias_index'].transform('min')
raw_mbfc['grouped_bias'] = raw_mbfc.apply(lambda x: bias_class[x['grouped_bias_index']], axis=1)
raw_mbfc['grouped_bias_diff'] = raw_mbfc.apply(lambda x: 10 * x['bias_index'] + x['grouped_bias_index'], axis=1)


group_bias_stat = raw_mbfc.groupby(['grouped_bias_diff'])['link'].agg('count')
print(group_bias_stat)
group_bias_array = np.zeros((9,9))
for ind, count in group_bias_stat.items():
    i = int(ind/10)
    j = ind%10
    group_bias_array[i][j] = count
group_bias_df= pd.DataFrame(group_bias_array)

plt.figure(figsize=(12,12))
sns.heatmap(data=group_bias_df, cmap="RdBu_r", xticklabels=bias_class, yticklabels=bias_class)

plt.xlabel('real bias', fontsize = 32)
plt.ylabel('transformed bias after group', fontsize = 32)
plt.show()



# to unify the format of mbfc and the integrated bias dataset
raw_mbfc['bias'] = raw_mbfc['grouped_bias']
raw_mbfc = raw_mbfc[['name','link','country','fact_reporting','media_type','traffic','press_freedom','cred_rank','bias']].reset_index()
raw_mbfc.drop_duplicates(subset ='link', keep='first', inplace=True)
raw_mbfc = pd.DataFrame(raw_mbfc)
raw_mbfc.to_csv("integrated_bias.csv")

