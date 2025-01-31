from points2regions import Points2Regions
import pandas as pd
# Example usage with a CSV file
data = pd.read_csv('data/pseudo_ct.csv')
idx = pd.read_csv("new_ct_indx.csv")
print(data.head())

# Create the clustering model
p2r = Points2Regions(
    data[['spotX', 'spotY']], 
    data['gene'], 
    pixel_width=1, 
    pixel_smoothing=5
)

# Cluster with a specified number of clusters
p2r.fit(num_clusters=32)
cluster_per_marker = p2r.predict(output='marker')
clusts =pd.Series(cluster_per_marker)
print(clusts.value_counts())

#print(idx)
col = idx.columns
#print(col)
lst_idx = idx[col].values.tolist()
#print(lst_idx)
lst_ind = [int(v[0]) for v in lst_idx]

cluster_per_marker = cluster_per_marker[lst_ind]

#spots
clusters_series= pd.Series(cluster_per_marker)
print(clusters_series.value_counts())


