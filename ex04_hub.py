#%%
import torch

#%%
# custom Model loading 
model = torch.hub.load('../yolov5','custom' ,'./md0611.pt',
                       source='local')
                    #    pretrained=True)

#%%
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
# %%
