from src.models.model import ImageCaptioningModel
import os
import torch

swin_ckpt = os.path.normpath("./checkpoint/feature_extraction/swinv2-large-patch4-window12to24-192to384-22kto1k-ft.pth")
yolo_ckpt = os.path.normpath("./checkpoint/feature_extraction/yolo11x-seg.pt")
rcnn_ckpt = os.path.normpath("./checkpoint/feature_extraction/fasterrcnn_resnet101.pth")
vocab_path = os.path.normpath("./checkpoint/vocab/vocab_Flickr8k.pkl")

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
