import pandas as pd
import glob

files =list(glob.glob("country_info/collections/*"))

df = pd.read_csv(files[0])

for i in range(len(files)):
    if i == 0:
        continue
    cur_df = pd.read_csv(files[i])
    df = df.append(cur_df)

df.to_csv("country_info/integrated_country.csv")