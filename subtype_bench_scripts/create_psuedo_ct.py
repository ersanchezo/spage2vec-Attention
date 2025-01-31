import os
import numpy as np
import pandas as pd
import random
SEED = 69
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(result_dir, exist_ok=True)
fig_dir = os.path.join(os.getcwd(), 'figures')
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv("data/spots_w_cell_segmentation_PCW6.5_2.csv")
#print(df.head(40))
print(df['gene'].value_counts()[34:])
print(df['gene'].value_counts()[:34])
print(df['parent_id'].value_counts().sum())
print(df['parent_id'].nunique())
#cnt_cells = df['parent_id'].value_counts().value_counts()
#print(cnt_cells.value_counts())
#artificial gene selection
cells_selected = random.sample(range(0,24705),124)
#genes to be used in the psuedo cell type:
new_ct = ["Syn Gene A","FAM210B","FAM46C"] 
index_changed = []
for c in cells_selected:
    new_genes = []
    for i in range(0,len(df[df.parent_id==c].index)):
        new_genes.append(new_ct[random.randint(0,2)])
    #df[df.parent_id==c]['gene']=new_genes
    df.loc[df.parent_id==c,'gene']=new_genes
    index_changed.extend(df.loc[df.parent_id==c].index.tolist())
#Add noise to the data set
print(df.size)
print(df.iloc[[15, 18, 25,400000]])
df.to_csv('data/pseudo_ct2.csv', index=False)
np.savetxt("new_ct_indx2.csv",
        index_changed,
        delimiter =",",
        fmt ='% s')


