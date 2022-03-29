
#coding: utf-8
# Imports
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import time
import copy
from collections import OrderedDict
from torch.autograd import Variable
output_size = 102

batch_size = {
                'train': 32,
                'test': 32,
                'valid':32
            }

input_sizes = {
                'vgg' : 25088,
                'densenet' : 1024,
                'alexnet' : 9216 ,
                'resnet': 512
            }

default_hidden_units = {
                'vgg': [ 4096],
                'densenet' : [1024],
                'alexnet' : [4608],
                'resnet' : [512]
            }

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

models = {
          'resnet': resnet18,
          'alexnet': alexnet,
          'vgg': vgg16,
          'densenet': densenet121
         }


data_transforms = {
    'train': transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                            ]),
    'test': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                            ]),
    'valid':transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                            ])
}

def load_data(data_dir):
    print("Func: load_data ")
    image_datasets = {val : datasets.ImageFolder(os.path.join(data_dir, val), data_transforms[val])
                      for val in ['train', 'valid', 'test']}

    data_loaders = {val :   torch.utils.data.DataLoader(image_datasets[val],
                            batch_size = batch_size[val], shuffle = True)
                      for val in ['train', 'valid', 'test']}

    datasets_sizes = {val :   len(image_datasets[val])
                      for val in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes
    return image_datasets, data_loaders, datasets_sizes, class_names

def build_classifier(input_size, output_size, hidden_layers, dropout_p):
    print("Func: build_classifier ")
    layers = OrderedDict()
    num_hidden_layer = 0
    if hidden_layers is not None:
        num_hidden_layer = len(hidden_layers)
    for i in range(num_hidden_layer):
        if i == 0:
            layers['fc' +  str(i)] = nn.Linear(input_size, hidden_layers[i])
        else:
            layers['fc' +  str(i)] = nn.Linear(hidden_layers[i -1], hidden_layers[i])
        layers['reLU' + str(i)] =  nn.ReLU()
        layers['dropout' + str(i)] =  nn.Dropout(p= dropout_p)

    if num_hidden_layer > 0:
        layers['fc' + str(num_hidden_layer) ] = nn.Linear(hidden_layers[ num_hidden_layer - 1 ], output_size)
    else:
        layers['fc' + str(num_hidden_layer) ] = nn.Linear(input_size, output_size)

    layers['output'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(layers)

    return classifier

def freeze_model_param(model):
    print("Func: freeze_model_param ")
    for param in model.parameters():
        param.requires_grad = False

def network(output_size, hidden_layers, dropout_p, learn_rate, arch, epochs,
            data_dir , trained , enable_GPU , save_dir):
        print("Func: network ")
        model = models[arch]
        freeze_model_param(model)
        input_size = input_sizes[arch]
        classifier =  build_classifier(input_size, output_size, hidden_layers, dropout_p)

        if arch == 'resnet':
            model.fc = classifier
            optimizer = optim.Adam(model.fc.parameters(), lr = learn_rate)
        else:
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)

        if enable_GPU == True and torch.cuda.is_available():
            device = "gpu"
            model = model.cuda()

        else:
            device = "cpu"

        print("Using: {}".format(device))
        if trained == False:
            criterion = nn.NLLLoss()
            if data_dir is not None:
                image_datasets, data_loaders, datasets_sizes, class_names = load_data(data_dir)
            else:
                print("Invalid Data_dir for training mode!")

            model = train(device, model, epochs, criterion, optimizer, data_loaders, datasets_sizes )

            test_accuracy(device, model, data_loaders['test'], datasets_sizes['test'])

            save_checkpoint(model, optimizer, input_size, output_size, hidden_layers,
                            image_datasets['train'], epochs, dropout_p, learn_rate,
                            arch,save_dir)

        return model, optimizer

def validation(device, model, validateloader, criterion):
    valid_loss = 0
    accuracy_count = 0

    for images, labels in validateloader:
        if device is 'gpu':
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        valid_loss += criterion(outputs, labels).item()* images.size(0)
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy_count += equality.type(torch.FloatTensor).sum()
    return valid_loss, accuracy_count

def train(device, model, epochs, criterion, optimizer, data_loader, data_sizes):
    print("Func: train ")
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())

    best_accuracy = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        print('*****' * 20)
        print("Epoch: {}/{}.. ".format(epoch+1, epochs))
        running_loss = 0.0
        for phase in ['train', 'valid']:
            print("Phase: {}".format(phase))
            running_corrects = 0
            if phase == 'train':
                model.train()

                for images, labels in data_loader['train']:
                    if device is 'gpu':
                        images = images.cuda()
                        labels = labels.cuda()
                    optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()* images.size(0)
            else:
                model.eval()
                valid_loss = 0
                accuracy = 0.0
                with torch.no_grad():
                    valid_loss, accuracy = validation(device, model, data_loader['valid'], criterion)
                epoch_loss = valid_loss / data_sizes['valid']
                epoch_accuracy = 100 * accuracy / data_sizes['valid']

                if  epoch_accuracy > 80 and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model = copy.deepcopy(model.state_dict())

                print("Training Loss: {:.4f}...".format(running_loss/ data_sizes['train']),
                  "Validation Loss: {:.4f}...".format(epoch_loss),
                  "Validation Accuracy: {:.4f}...".format(epoch_accuracy))
                running_loss = 0
    print('*****' * 20)
    time_elapsed = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60 ))
    if best_accuracy is not 0.0:
        print("Best Accuracy achieved: {:.4f}".format(best_accuracy))
        model.load_state_dict(best_model)
    print('*****' * 20)
    return model

def save_checkpoint(model, optimizer, input_size, output_size, hidden_layers,
                    train_datasets, epochs, dropout, learn_rate, arch, save_dir = None):
    print("Func: save_checkpoint ")
    checkpoint = {
                'input_size': input_size,
                'output_size': output_size,
                'hidden_layers': [size for size in hidden_layers],
                'state_dict': copy.deepcopy(model.state_dict()),
                'class_to_idx':train_datasets.class_to_idx,
                'epochs':epochs,
                'optimizer_state':copy.deepcopy(optimizer.state_dict()),
                'dropout':dropout,
                'learn_rate':learn_rate,
                'arch': arch
             }
    if save_dir is None:
        path = str(arch) + '_checkpoint.pth'
    else:
        path = save_dir

    torch.save(checkpoint,(path))
    print("checkpoint saved as {}".format(path))

def load_checkpoint(checkpoint_file_path, enable_GPU):
    print("Func: load_checkpoint ")
    checkpoint = torch.load(checkpoint_file_path)
    model, optimizer = network(
                               checkpoint['output_size'],
                               checkpoint['hidden_layers'],
                               checkpoint['dropout'],
                               checkpoint['learn_rate'],
                               checkpoint['arch'],
                               checkpoint['epochs'],
                               None,
                               True,
                               enable_GPU,
                               None
                              )
    model.load_state_dict( checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer

def process_image(image_path):
    print("Func: process_image ")
    preprocess = data_transforms['test']
    pil_image = Image.open(image_path)
    image_tensor = preprocess(pil_image)
    np_image = np.array(image_tensor)
    np_image = np.clip(np_image, 0, 1)
    return np_image, image_tensor

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def index_to_name(json_file_name):
    with open(json_file_name, 'r') as f:
        index_to_names = json.load(f)
    return index_to_names

def predict(image_path, model,  enable_gpu, top_k=5):
    print("Func: predict ")
    image ,image_tensor = process_image(image_path)
    image_tensor.unsqueeze_(0)
    if enable_gpu == True:
        model = model.cuda()
        image_tensor = image_tensor.cuda()
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    probs =  torch.exp(outputs)
    ps, index = probs.topk(top_k)
    return ps, index

def display_result(image_tensor, classes, probs):

    f, axarr = plt.subplots(2, 1)
    image_tensor = image_tensor.transpose((1, 2, 0))
    axarr[0].imshow(image_tensor)
    axarr[0].set_title('Sample Image')
    y_pos = np.arange(len(classes))

    axarr[1].barh(y_pos, probs, align = 'center', color = 'red')
    axarr[1].set_yticks(y_pos)
    axarr[1].set_yticklabels(classes)
    _ = axarr[1].set_xlabel("Probabilities")

def test_accuracy(device, model, test_loader, test_size):
    print("Func: test_accuracy ")
    accuracy_count = 0
    model.eval()
    for images, labels in test_loader:
        if device == 'gpu':
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy_count += equality.type(torch.FloatTensor).sum()
    print(accuracy_count)
    accuracy = accuracy_count * 100 / test_size
    print("Test accuracy: {:.2f}".format(accuracy))

def inference(input_file_path, checkpoint, top_k, category_names, enable_gpu):
    print("Func: inference ")
    model, optimizer = load_checkpoint(checkpoint, enable_gpu)
    if enable_gpu == True and torch.cuda.is_available():
     enable_gpu = True
    else:
     enable_gpu = False

    ps, index = predict(input_file_path, model, enable_gpu,  top_k = top_k)

    index = np.array(index[0])

    ps = np.array(ps[0])

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in index]
    print("{} Top {} Classes {}".format('*****' * 7 , top_k , '*****' * 7,))
    for prob, class_index in zip(ps, top_classes):
        if category_names is not None:
            index_to_names = index_to_name(category_names)
            print("Probabilities: {:.4f}... | Class: {:3}  | Class Name: {:30} |".format(prob, class_index, index_to_names[class_index]))
        else:
            print("Probabilities: {:.4f}... | Class: {:3} |".format(prob, class_index))
    print('*****' * 17)
