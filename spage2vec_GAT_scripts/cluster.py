import os
import random
import time
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

sc.logging.print_header()
sc.settings.verbosity = 4

data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')

model = 'SAGE'
filename = 'SAGE-embedding-20240219_pseudo2_final.npy'
node_emb = np.load(os.path.join(result_dir,filename))
print(filename)
print(node_emb.shape)
df_nodes =pd.read_csv(os.path.join(result_dir, 'nodes_pseudo2_kd.csv'),index_col=0)
df_heart = pd.read_csv(os.path.join(data_dir, 'pseudo_ct2.csv'))
df_heart = df_heart.loc[df_nodes.index]
adata = sc.AnnData(X = np.copy(node_emb), obs = df_heart)

#clustering
sc.pp.neighbors(adata, n_neighbors=15, random_state = 42)
adata_name = time.strftime('{}-%Y%m%d_pseudo2.h5ad'.format(model))
adata.write(os.path.join(result_dir, adata_name))
# 0.8 for GAT embedding 32 clusters
sc.tl.leiden(adata, resolution=0.20, random_state=42)
adata.write(os.path.join(result_dir, adata_name))
print(adata)

#umap
sc.tl.umap(adata, min_dist = 0.5, n_components = 2, random_state = 42)
file_name = time.strftime('{}-umap-%Y%m%d_pseudo2_.npy'.format(model))
np.save(os.path.join(result_dir, file_name), adata.obsm['X_umap'])
print(file_name)
