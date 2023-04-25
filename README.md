# My first kaggle competition
## Learning how kaggle competitions work with facial keypoints detection





| Phase 1: EDA | Phase 2: Exploring Architectures| Phase 3: Model Improvement | Phase 4: Future Direction | 
| ------------- | ------------- |------------- | ------------- |
| Display an image with all 30 key points (notebook 1) | Minimum Viable Product Model (notebook 1) | Gradient Accumulation (notebook 2.1) | TTA (test time augmentation)
| Create train/test/val splits (mininotebook 1+2) | Simple CNN class (notebook 2) | Optimal LR / hyperparameters (notebook 2.2) | Bagging Multiple Models
| Create pytorch dataloaders (notebook 2) | Resnet18 / Resnet50 (notebook 3) | Freezing layers with pretrained models (notebook 3) | L2 Regularization / Weight decay
| Device agnostic code (GPU/CPU) | -- | Regularization w/ dropout (notebook 4) (first layer, last layer, varying dropout likelihood) |




Notes: 
Used a small subset of data (40 images) for rapid iteration in phase 1 and 2
Ran on low memory gpu (personal computer) so needed to include gradient accumilation and employ device agnostic code
