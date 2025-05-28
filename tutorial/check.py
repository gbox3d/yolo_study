#%%
import torch, torchvision, ultralytics
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torchvision.ops.nms)         # Capsule object … 나오면 OK
ultralytics.checks()               # 의존성 점검, CUDA/GPU OK 메시지 확인

# %%
