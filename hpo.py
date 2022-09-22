import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, epoch, device, criterion):

    logger.info(f"Epoch test dataset:{epoch}")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_loss += loss.item() * inputs.size(0) 
            correct += pred.eq(labels.view_as(pred)).sum().item() 
            
        total_loss = test_loss / len(test_loader.dataset)
        total_acc = correct/ len(test_loader.dataset)
        
        logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, correct, len(test_loader.dataset), 100.0 * total_acc / len(test_loader.dataset)
            )
        )

        
def train(model, train_loader, criterion, optimizer, epoch, device):
    logger.info(f"Epochs train dataset: {epoch}")
    model.train()
    train_loss = 0
    correct = 0
    running_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)  
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1,  keepdim=True)
        train_loss += loss.item() * inputs.size(0) 
        correct += pred.eq(labels.view_as(pred)).sum().item()
        running_samples+=len(inputs)
        loss.backward()
        optimizer.step()
        if running_samples % 500  == 0:
            accuracy = (correct/running_samples)
            logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                    running_samples,
                    len(train_loader.dataset),
                    100.0 * (running_samples / len(train_loader.dataset)),
                    loss.item(),
                    correct,
                    running_samples,
                    100.0*accuracy
                )
            )
    total_loss = train_loss / len(train_loader.dataset)
    total_acc = correct/ len(train_loader.dataset)
    logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        total_loss, correct, len(train_loader.dataset), 100.0 * total_acc
    ))   
    return model
        
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   
        
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                             nn.Linear(num_features, 256), 
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True)
                            )
        
    return model

def create_data_loaders(data, batch_size):
    
    train_dataset_path = os.path.join(data, "train")
    test_dataset_path = os.path.join(data, "test")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size )
    
    return train_data_loader, test_data_loader

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
        
        
    model=net()
    model = model.to(device)
    
    
    train_data_loader, test_data_loader = create_data_loaders(args.data_dir, args.batch_size)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr,  eps= args.eps)
    
    for epoch in range(1, args.epochs+1):
        logger.info(f"Epoch Training: {epoch}")
        model=train(model, train_data_loader, loss_criterion, optimizer, epoch, device)
        logger.info(f"Epoch Testing: {epoch}")
        test(model, test_data_loader, epoch, device, loss_criterion)
        
    logger.info("Saving Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Image Classification using AWS SageMaker")
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
        
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
        
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 1.0)"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        metavar="EPS",
        help="eps (default: 1e-8)" )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.environ['SM_CHANNEL_TRAINING']
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.environ['SM_MODEL_DIR']
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    args=parser.parse_args()
    
    
    main(args)
