from efficientnet_pytorch import EfficientNet

from SETTINGS import DEVICE

model = EfficientNet.from_pretrained('efficientnet-b5').eval().to(DEVICE)
