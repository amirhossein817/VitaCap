# VitaCap: Transformer-based Image Captioning

VitaCap is a Transformer-based image captioning model that enhances multi-vision attention features using state-of-the-art feature extraction techniques and an innovative feature fusion module. This repository is designed to generate accurate and meaningful captions for images by leveraging hierarchical, object-level, and pixel-level visual features.

---

## **Features**
- **Feature Extraction:**
  - **Swin Transformer:** Extracts hierarchical grid-level features.
  - **Faster R-CNN:** Provides object-level semantic features.
  - **YOLO:** Captures pixel-level details for fine-grained representations.
- **Feature Fusion Module:**
  - Combines extracted features using two Multi-Head Attention (MHA) layers.
  - Outputs a fused representation through Feed Forward Networks (FFN) and Add & Norm layers.
- **Transformer Encoder-Decoder Architecture:**
  - Encodes fused features into a rich latent space.
  - Decodes captions using masked multi-head attention mechanisms.
- **Training Pipeline:**
  - Configurable training with support for pre-trained models.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/amirhossein817/VitaCap.git
   cd VitaCap
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up pretrained models (e.g., Swin, Faster R-CNN, YOLO) as per instructions in their respective directories.

---

## **Usage**
### **Preprocessing Annotations**
Run the preprocessing script to tokenize and build the vocabulary for annotations:
```bash
python src/data/build_vocab.py --input_file <path_to_annotations> --output_file vocab.pkl
```

### **Training the Model**
Start the training pipeline with:
```bash
python train/train.py --config_file <path_to_config>
```
Key configuration parameters:
- `batch_size`
- `learning_rate`
- `num_epochs`
- `feature_extraction_method`

### **Inference**
Generate captions for images using:
```bash
python inference.py --image <path_to_image> --model_checkpoint <path_to_checkpoint>
```

---

## **File Structure**
```
├─ src
│   ├─ models
│   │   ├─ faaster_rcnn_featuure_extractor.py   # Object features extraction
│   │   ├─ swin_feature_extractor.py           # Grid features extraction
│   │   ├─ swin_transformer.py                 # Swin Transformer architecture
│   │   ├─ yolo_feature_extraction.py          # Pixel features extraction
│   │   ├─ fusion_module.py                    # Feature fusion logic
│   │   ├─ encoder.py                          # Transformer encoder
│   │   ├─ decoder.py                          # Transformer decoder
│   │   ├─ model.py                            # Model logic
│   ├─ data
│   │   ├─ build_vocab.py                      # Annotation preprocessing
├─ train
│   ├─ train.py                                # Training pipeline
```

---

## **Technical Details**
### **Feature Fusion Module**
The fusion module combines visual features using:
1. **Dual Multi-Head Attention (MHA):**
   - One uses Feature 1 as Query and Feature 2 as Key and Value.
   - The other reverses this arrangement.
2. **Feed Forward Networks (FFN):** Processes the MHA outputs.
3. **Add & Norm:** Combines and normalizes the results for stability.

### **Loss Function**
- Cross-entropy loss for caption generation.

### **Evaluation Metrics**
- BLEU
- CIDEr
- ROUGE

---

## **Future Work**
- Extend support for additional feature extractors.
- Implement reinforcement learning techniques (e.g., CIDEr optimization).
- Provide pre-trained checkpoints for faster deployment.

---

## **Acknowledgements**
This project utilizes:
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn)
- [YOLO](https://github.com/AlexeyAB/darknet)

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**
For inquiries, reach out to [Amir Hossein](mailto:amirhossein817@example.com).
