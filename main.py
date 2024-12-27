from src.models.model import ImageCaptioningModel
import os
import torch

swin_ckpt = os.path.normpath("./checkpoint/feature_extraction")
yolo_ckpt = os.path.normpath("./checkpoint/feature_extraction")
rcnn_ckpt = os.path.normpath("./checkpoint/feature_extraction")
vocab_path = os.path.normpath("./checkpoint/feature_extraction")

# Instantiate model
model = ImageCaptioningModel(swin_ckpt, yolo_ckpt, rcnn_ckpt, vocab_path)

# Dummy input image (batch_size=1, 3 channels, 384x384 resolution)
dummy_image = torch.randn(1, 3, 384, 384)

# Forward pass
logits = model(dummy_image)
print("Caption Logits Shape:", logits.shape)

# Generate predicted caption
caption = model.predict_caption(logits)
print("Predicted Caption:", " ".join(caption))
