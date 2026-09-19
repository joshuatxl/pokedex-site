# Pokedex site

## Project background and site link

A convolutional neural network that identifies which of the original 151 Pokemon appears in a photo. The neural network model is a reworked version of an assignment as part UNSW's ZZEN9444 Neural Networks, Deep Learning course. 

Access the Pokedex site here: https://joshuatxl.github.io/pokedex-site/


## How to use it

Upload a photo (preferrably in .png or .jpg) showing one of the original 151 Pokemon, and the model will return its best guess of what Pokemon is in the image along with a confidence score.


## Data source

Images of each of the original Pokemon lineup, with between 30 to 60 images for each of the 151 classes: https://www.kaggle.com/datasets/thedagger/pokemon-generation-one

Images span a diverse mix of image types (game sprites, official artwork, images from trading cards, images from the cartoon, fan art).

Gaps: 
- Images of Nidoran's female and male forms were missing, which needed to be manually collected. 


## Model, training, prediction, site specifications

| Stage | Details |
|---|---|
| **Model** | Fully fine-tuned ResNet-50 architecture with pre-trained weights; input of 256 × 256 pixels. |
| **Training** | PyTorch. Class-weighted loss + square-root-weighted oversampling to counter a ~7.5× class imbalance; heavy augmentation (random crop/flip/rotation/color jitter/TrivialAugment); AdamW with cosine LR decay. Model was fine-tuned with a training-to-validation split of 0.8 for 30 epochs, with the final model trained with the full dataset (training-to-validation split of 1.0) at 20 epochs. The decision to run the final model for 20 epochs was due to compute constraints of Google Colab's free tier and that test accuracy plateaus well before epoch 20 in during finetuning.|
| **Prediction** | Trained model is exported to ONNX, run in-browser via `onnxruntime-web`. |
| **Site** | Static HTML/CSS/JS hosted on GitHub Pages. |


## Results

Final fine-tuning run:
- Train accuracy: 96%
- Test accuracy: 88%

Final training run:
- Accuracy: 96%

Gaps:
- Consistent overfitting of 6-8%% during fine-tuning training runs. 


## Major changes from the base neural network model:

Various aspects of the model submitted as part of the course assignment was reworked to improve functionality: 
- ResNet-50 architecture was used instead of RegNetY_1.6GF, which was originally implemented to comply with model-size constraints.
- Implemented inverse-frequency class weighting and a sampler that oversamples classes with comparatively fewer input images within each epoch to handle class imbalance during training.
- Applied 'transforms.Resize(288)' to rescale the test dataset images' shorter size to 288, preserving the aspect ratio.
- Applied 'transforms.CenterCrop(256)' to crop the image to the size the model was trained on.


## Limitations

- Model is trained to identify only one Pokemon in an image, and will silently select the Pokemon with the highest confidence score if more than one Pokemon is present in an image.

