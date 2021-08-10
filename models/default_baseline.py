from torchvision.models import resnext101_32x8d

from SETTINGS import DEVICE

model = resnext101_32x8d(pretrained=True, progress=True)
model = model.eval().to(DEVICE)
