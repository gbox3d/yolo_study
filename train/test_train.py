#%%
dataset_path = '/home/gbox3d/work/dataset/pistol/'

with open(f'{dataset_path}val.txt', 'r') as f:
  val_list = (f.read()).split('\n')
  print(val_list[5])
# %%
