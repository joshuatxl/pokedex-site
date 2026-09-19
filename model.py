#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import WeightedRandomSampler

from config import device



#=============================================================================
# Metaparameters and training options           
#=============================================================================
dataset = "/content/dataset"
train_val_split = 1
batch_size = 40
epochs = 20

# Define mean and std of ImageNet database, evaluation resize and crop resolution
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
eval_resize = 288
eval_crop = 256

def list_classes(dataset_root):
    return sorted(d for d in os.listdir(dataset_root)
                  if os.path.isdir(os.path.join(dataset_root, d)))

# Inverse-frequency class weights to handle class imbalance
def _class_weights(dataset_root):
    classes = list_classes(dataset_root)
    counts = torch.tensor(
        [len(os.listdir(os.path.join(dataset_root, c))) for c in classes],
        dtype=torch.float)
    return counts.sum() / (len(counts) * counts)   # normalised inverse frequency

class_weights = _class_weights(dataset).to(device)

# Sampler that oversamples classes with comparatively fewer input images within each epoch; weighted by square-root inverse frequency rather than full
# inverse frequency so that the small number of inputs of these classes aren't repeated too frequently, leading to overfit
def make_train_sampler(subset):
    base = subset.dataset
    targets = [base.targets[i] for i in subset.indices]
    class_counts = torch.bincount(torch.tensor(targets), minlength=len(base.classes)).float()
    sample_weights = 1.0 / torch.sqrt(class_counts[torch.tensor(targets)])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


#=============================================================================
# Transformations
#=============================================================================
def transform(mode):

    #Normalise the input images to the ImageNet mean and standard deviation; required as pre-trained weights are used
    normalise = transforms.Normalize(mean = norm_mean, std = norm_std)

    if mode == 'train':
        return transforms.Compose(
            [
                #Randomly crops the input images between 70% and 100% its area, then resize the image
                transforms.RandomResizedCrop(256, scale = (0.7, 1.0)),
                #Randomly flips the input images with a 50% chance
                transforms.RandomHorizontalFlip(),
                #Randomly rotate the input images from -15 to 15 degrees clockwise or anticlockwise
                transforms.RandomRotation(degrees = 15),
                #Randomly apply translation to the input images
                transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
                #Randomly apply an augmentation operation (e.g., rotation, brightness, etc.) to varying extents
                transforms.TrivialAugmentWide(),
                #Apply mild color jitter to simulate different lighting conditions in images
                transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.1),
                #Convert the input image to a tensor
                transforms.ToTensor(),
                #Apply normalisation
                normalise
            ]
        )

    
    elif mode == 'test':
        return transforms.Compose(
            [
            #Rescale the test dataset images' shorter size to 288 pixels, preserving the aspect ratio.
            transforms.Resize(eval_resize),
            #Crop the image to the size the model was trained on (256 pixels).
            transforms.CenterCrop(eval_crop),
            transforms.ToTensor(),        
            normalise                     
            ]
        )


#=============================================================================
# Define the Module to process the images and produce labels
#=============================================================================
class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #Use the ResNet50 architecture with pre-trained weights
        model_backbone = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)

        #Initialise a new classifier with 151 output nodes, one for Pokemon
        num_features = model_backbone.fc.in_features
        model_backbone.fc = nn.Sequential(
            #Implement dropout to randomly turn off 30% of the input features to the classifier during each forward pass in training
            nn.Dropout(p = 0.3),
            nn.Linear(num_features, 151)
        )

        self.model = model_backbone

    def forward(self, input):
        return self.model(input)

net = Network()
    
#=============================================================================
# Specify the optimizer and loss function               
#=============================================================================
#Implement AdamW as an optimiser
optimizer = optim.AdamW(
    net.parameters(),
    #Initialise a small learning rate for precise convergence
    lr = 0.0001,
    #Implement regularisation to penalise large weights and reduce overfitting
    weight_decay = 0.01
    )

#Implement Cross Entropy Loss as the loss function, which evaluates how well the model's predicted class probabilities match
# true labels for multi-class classification; weight by inverse class frequency to counter class imbalance, and implement
# label smoothing to discourage the model from being trained to overconfidently predict target classes
loss_func = nn.CrossEntropyLoss(weight = class_weights, label_smoothing = 0.1)


#=============================================================================
#Custom weight initialization and lr scheduling are optional 
#=============================================================================

def weights_init(m):
    return

#Implement cosine annealing to decay the learning rate to 0 gradually across epochs during a training run for more precise convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= epochs)


