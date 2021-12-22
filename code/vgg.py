import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
from torchvision import models

import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from daze import plot_confusion_matrix
from log import log

# H Y P E R P A R A M E T E R S

#representing the emotions to classify
num_classes = 7
batch_size = 40
#number of training iterations
num_epochs = 100
learning_rate = 0.001
#path to data
root_dir = r"<INSERT PATH HERE>"

# setting the device on which the modell will train :)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# to improve performace images are going to get normalized, meaning each of their colour-channels value [0, 255] will be devided by 255
transforms = transforms.Compose([
                                # will transform the image to tensor, making it hereby applicable in ML context
                                ToTensor(),
                                # values are related to the model used for transfer learning which was trained on the ImageNet dataset 
                                # standardization = (pixelValue - mean) / stdDev for each channel
                                # NOTE: these values apply for PyTorch ImageNetModels ONLY -> see: https://pytorch.org/vision/stable/models.html
                                Normalize(  
                                    # 1st tuple = mean of each channel for all images the model was trained on
                                    [0.485, 0.456, 0.406], 
                                    # 2nd tuple = std-deviation of each channel for all images the model was trained on
                                    [0.229, 0.224, 0.225])
                                ]) 

# prepare data to be labeled by assigning path and applying normalization on image data
training_data = ImageFolder(root = os.path.join(root_dir, 'training'), transform=transforms)  
validation_data = ImageFolder(root = os.path.join(root_dir, 'validation'), transform=transforms)    
test_data = ImageFolder(root = os.path.join(root_dir, 'test'), transform=transforms)    

#create labeled datasets
training_dataloader = DataLoader(dataset = training_data, batch_size = batch_size, shuffle = True)
validation_dataloader = DataLoader(dataset = validation_data, shuffle = True)
test_dataloader = DataLoader(dataset = test_data, shuffle = True)

# this class will be used to overwrite the avgPooling Layer inside the pretrained PyTorch model and simply pass on whatever value is given to it  
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# loading a pretrained model
model = models.vgg16(pretrained=False)

# simulate Transfer Learning by restricting the adjustment of weights & biases up to the point, where the modified block beginns
for param in model.parameters():
    param.requires_grad = False

# Modifying the dense part of the CNN 
# 'deactivating' the avgPooling
model.avgpool = Identity() 
#modified dense block, diverging from the original, especially regarding the number of output classes (in original VGG16 = 1000)
#for entire model architecture see Appendix
model.classifier = nn.Sequential(
                                # 25088 = (224*224) / 2 -> due to kernel size and stride of the maxPooling layer
                                nn.Linear(25088, 500), 
                                nn.ReLU(),
                                nn.Linear(500, num_classes),
                                nn.LogSoftmax(dim=1)                        
                                )                                    

# loading model to CPU or GPU if available
model.to(device)
# defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# list to store confusion matrices for visualising the progress during validation phase
cms_val = []
# list to store accuracy during training
acc_train = []
for epoch in range(num_epochs):

# T R A I N I N G

    #lists to keep track of the models performance
    target_list = []
    prediction_list=[]

    # setting the model to training mode, so weights & biases will be adjusted within this phase
    model.train()    
    for batch_index, (data, targets) in enumerate(training_dataloader):
        # loading training data to GPU if available, else CPU
        data = data.to(device = device)
        targets = targets.to(device = device)
        # resetting the gradient @ every batch
        optimizer.zero_grad()
        # forward pass
        prediction = model(data)
        # calculating loss
        loss = criterion(prediction, targets)
        # backpropagating the loss through the network
        loss.backward()
        # gradient descent, approximating minimum of cost function, updating weights and biases accordingly
        optimizer.step()
        #storing results for evaluating the training process
        target_list.append(targets.cpu().detach().numpy())
        prediction_list.append(prediction.argmax(dim=1).cpu().detach().numpy())
    #concatenation is necessary due to sklearns accuracy_score() won't handle nested lists
    acc_train.append(accuracy_score(list(np.concatenate(target_list)), list(np.concatenate(prediction_list)))*100)

# V A L I D A T I O N

    #lists to keep track of the models performance
    target_list = []
    prediction_list=[]
    
    # setting the model to testing mode, weights & biases remain fixed
    model.eval()
    for batch_index, (data, targets) in enumerate(validation_dataloader):
        data = data.to(device = device)
        targets = targets.to(device = device)
        optimizer.zero_grad()
        pred_on_val_data = model(data)
        #store true labels and prediction for evalution
        target_list.append(targets.cpu().detach().numpy())
        prediction_list.append(pred_on_val_data.argmax(dim=1).cpu().detach().numpy())
    # store results of validation epochs in list (confusion matrix objects)
    cms_val.append(confusion_matrix(target_list, prediction_list))

# save model to hard drive      
# torch.save(model.state_dict(), "trained_Model.pth")
print("============= FINISHED TRAINING =============")

# T E S T I N G

# works similar to validation
target_list=[]
prediction_list=[]
for batch_index, (data, targets) in enumerate(test_dataloader):
    data = data.to(device = device)
    targets = targets.to(device = device)
    optimizer.zero_grad()
    pred_on_val_data = model(data)
    target_list.append(targets.cpu().detach().numpy())
    prediction_list.append(pred_on_val_data.argmax(dim=1).cpu().detach().numpy().item())

print("============= FINISHED TESTING =============")    

# P L O T S / L O G S

# print evaluation of test-set as confusion matrix, using Daze-library (wrapper for sklearn)
# src: https://pypi.org/project/daze/
plt.figure(figsize=(9,9))
plot_confusion_matrix(
    confusion_matrix(target_list, prediction_list), 
    display_labels=training_data.class_to_idx.keys(), 
    measures=('fpr','tnr', 'fnr', 'a', 'p', 'r', 'f1', 'c'),
    xticks_rotation=45,
    measures_format='0.3f',
    colorbar=None,
    normalize='true')
exportPath= os.path.join(r'{}'.format(root_dir), ("cm"+"_"+root_dir[16:].replace("\\", "_")))
plt.savefig(exportPath, bbox_inches='tight', pad_inches=0.0)
plt.close()

# Creating plots based on training & validation values over epochs

prec=[]
recall=[]
spec=[]
acc_val=[]
#src: https://stackoverflow.com/questions/48100173/how-to-get-precision-recall-and-f-measure-from-confusion-matrix-in-python
for e in range(num_epochs):
    # evaluating the validation phase
    TP = np.diag(cms_val[e])
    FP = np.sum(cms_val[e], axis=0) - TP
    FN = np.sum(cms_val[e], axis=1) - TP
    TN = []
    for i in range(num_classes):
        temp = np.delete(cms_val[e], i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))

    prec.append((np.mean((TP/(TP+FP))))*100.0)
    recall.append((np.mean((TP/(TP+FN))))*100.0)
    spec.append((np.mean((TN/(TN+FP))))*100.0)
    acc_val.append((np.mean((TP+TN)/(TP+FP+FN+TN)))*100.0)

print(acc_train)
plt.xlabel('Epochs')
plt.ylabel('Percentage')
plt.title('Learning Progress')

plt.plot(range(num_epochs), acc_train, c = 'blue', label="Training Accuracy", ls = "--", linewidth= 1.0)
plt.plot(range(num_epochs), acc_val, c='red', label="Validation Accuracy", ls = "--", linewidth= 1.0)
plt.plot(range(num_epochs), prec, c='lime', label="Avg. Val. Precision", linewidth= 1.0)
plt.plot(range(num_epochs), recall, c='darkorange', label="Avg. Val. Recall", linewidth= 1.0)
plt.plot(range(num_epochs), spec, c='teal', label="Avg. Val. Specificity", linewidth= 1.0)
plt.legend()

exportPath= os.path.join(r'{}'.format(root_dir), ("Eval"+"_"+root_dir[16:].replace("\\", "_")))
plt.savefig(exportPath, bbox_inches='tight', pad_inches=0.0)
plt.close()

#uncomment if numeric log is desired
"""
log(
    target_list, 
    prediction_list, 
    training_data.class_to_idx, 
    num_classes,
    os.path.join(r'{}'.format(root_dir), ("log"+"_"+root_dir[16:].replace("\\", "_") + ".txt"))
)
"""