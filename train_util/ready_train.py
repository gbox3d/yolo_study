#%%
#config data 
import yaml
with open(f'test_conf.yaml', 'r') as f:
  config_data = yaml.load(f)

print(config_data)

#%%
#총이미지 갯수 카운트 
dataset_path = config_data['dataset_path']

from glob import glob
img_list = glob(f'{dataset_path}export/images/*.jpg')
print( f'total : {len(img_list)}')
# %%
#이미지 분류 
from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)
print(f' train : {len(train_img_list)}, test : {len(val_img_list)}')

# %%
#save labeling datas
with open(f'{dataset_path}train.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open(f'{dataset_path}val.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')

print('save done')
# %%
#yaml 데이터 수정  
import yaml

with open(f'{dataset_path}data.yaml', 'r') as f:
  data = yaml.load(f)

print(data)

#%%

data['train'] = f'{dataset_path}train.txt'
data['val'] = f'{dataset_path}val.txt'

with open(f'{dataset_path}data.yaml', 'w') as f:
  yaml.dump(data, f)

print(data)
# %%
