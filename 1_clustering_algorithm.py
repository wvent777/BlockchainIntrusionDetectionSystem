import warnings
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

# LOADING DATA from ids_data_py (for.py files)
df = pd.read_parquet("./ids_data_py/cic-collection.parquet")

# Drop unnecessary columns (Extra Label Column)
df.drop(columns=["Label"], axis=1, inplace=True)

# Generate improved CIC Collection to compare

df_improved = df.drop(columns=['PSH Flag Count', 'ECE Flag Count',
                               'RST Flag Count', 'ACK Flag Count',
                               'Fwd Packet Length Min', 'Bwd Packet Length Min',
                               'Packet Length Min', 'Protocol', 'Down/Up Ratio'],
                      axis=0)
df_improved = df_improved.drop(columns=['Bwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
                                        'Bwd Avg Packets/Bulk', 'Bwd PSH Flags',
                                        'Bwd URG Flags', 'CWE Flag Count',
                                        'FIN Flag Count', 'Fwd Avg Bulk Rate',
                                        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                                        'Fwd URG Flags'])

# Export parquet file to .py data directory
df_improved.to_parquet('./ids_data_py/cic-collection-improved.parquet')

# PREPROCESSING
# Z-Score Normalization
# Original Data
features = df.dtypes[df.dtypes != 'object'].index
df[features] = df[features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# Fill nan values with 0
df = df.fillna(0)

# Improved Data
features_1 = df_improved.dtypes[df_improved.dtypes != 'object'].index
df_improved[features_1] = df_improved[features_1].apply(
    lambda x: (x - x.mean()) / (x.std()))
# Fill nan values with 0
df_improved = df_improved.fillna(0)

# Encoding Labels of Original Data
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

print("Original Data Class Distribution")
print(df.ClassLabel.value_counts(), end="\n\n")

# Encoding Labels of Improved Data
df_improved.iloc[:, -1] = labelencoder.fit_transform(df_improved.iloc[:, -1])
df_improved.ClassLabel.value_counts()

# DATA SAMPLING
# Can Adjust Sample Size later
# HDBSCAN was taking too long with the original data set
# Going to resample twice to get a smaller data set

# Original Data
df_sample1 = df.sample(frac=0.005, random_state=1)
print(f"DF Sampled Shape: {df_sample1.shape} \n")

df_minor = df_sample1[(df_sample1['ClassLabel'] == 5) | \
                      (df_sample1['ClassLabel'] == 7) | (df_sample1['ClassLabel'] == 6)]
df_major = df_sample1.drop(df_minor.index)

X = df_major.drop(['ClassLabel'], axis=1)
y = df_major.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

# Data Sampling the original dataset
# Use HDBSCAN to Cluster the data samples
print('start clustering original data')
clusterer = hdbscan.HDBSCAN()
clusterer.fit(X)
cluster_labels = clusterer.labels_
df_major["ClusterLabels"] = cluster_labels
print(df_major["ClusterLabels"].value_counts())
print("done clustering original data \n")

cols = list(df_major)
# with 2 layer of metadata removed it is 58, without it is 79
cols.insert(79, cols.pop(cols.index('ClassLabel')))
df_major = df_major.loc[:, cols]


def sampling(df):
    name = df.name
    frac = 1.0
    return df.sample(frac=frac)


result = df_major.groupby('ClusterLabels', group_keys=False).apply(sampling)
result = result.drop(["ClusterLabels"], axis=1)
result = result.append(df_minor)
print("Original Data Sampled Shape: ", result.shape)
print("Original Data Sampled ClassLabel Counts: \n", result.ClassLabel.value_counts(), end="\n\n")

# export the original data sample to csv file
result.to_csv('./ids_data_py/CIC_Collection_clean_sample.csv', index=0)

# Sampling from improved data
# can change the sampling size later
df_improved_sample1 = df_improved.sample(frac=0.005, random_state=1)
print(f"DF Improved Sampled Shape: {df_improved_sample1.shape}")

# Minors are 5: Infiltration, 7: Webattack, 6: Portscan
# Keep the minor size and sampling from the remaining major classes
dfi_minor = df_improved_sample1[(df_improved_sample1['ClassLabel'] == 5) | \
                                (df_improved_sample1['ClassLabel'] == 7) | \
                                (df_improved_sample1['ClassLabel'] == 6)]
dfi_major = df_improved_sample1.drop(dfi_minor.index)

X_im = dfi_major.drop(['ClassLabel'], axis=1)
y_im = dfi_major.iloc[:, -1].values.reshape(-1, 1)
y_im = np.ravel(y_im)

# Data Sampling the improved dataset
# Use HDBSCAN to Cluster the data samples
print('start clustering improved data')
clusterer = hdbscan.HDBSCAN()
clusterer.fit(X_im)
cluster_labels = clusterer.labels_
dfi_major["ClusterLabels"] = cluster_labels
print(dfi_major["ClusterLabels"].value_counts())
print("done clustering improved data \n")

cols_im = list(dfi_major)
# with 2 layer of metadata removed it is 58, without it is 69
cols_im.insert(58, cols_im.pop(cols_im.index('ClassLabel')))
dfi_major = dfi_major.loc[:, cols_im]

result_im = dfi_major.groupby('ClusterLabels', group_keys=False).apply(sampling)
result_im = result_im.drop(["ClusterLabels"], axis=1)
result_im = result_im.append(dfi_minor)

print("Improved Data Sampled Shape: ", result_im.shape)
print("Improved Data Sampled ClassLabel Counts: \n", result_im.ClassLabel.value_counts(), end="\n\n")

# Export the improved data sample to csv
result_im.to_csv('./ids_data_py/CIC_Collection_improved_sample.csv', index=0)
