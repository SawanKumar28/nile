import pandas as pd
import os
import sys

esnli_path = sys.argv[1]

#Merge train files
merged_train_file_path = os.path.join(esnli_path, "dataset", "esnli_train.csv")
train1_path = os.path.join(esnli_path, "dataset", "esnli_train_1.csv")
train2_path = os.path.join(esnli_path, "dataset", "esnli_train_2.csv")
d1 = pd.read_csv(train1_path, index_col="pairID")
d2 = pd.read_csv(train2_path, index_col="pairID")

d = pd.concat([d1, d2], 0, sort=False)
d_nna = d.dropna()

print ("Merging train files with lengths ", len(d1), len(d2), ", output length", len(d_nna))
d_nna.to_csv(merged_train_file_path)
