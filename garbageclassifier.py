import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from torch.multiprocessing import freeze_support
from PIL import Image
from pathlib import Path
import dotenv


dotenv.load_dotenv()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def calculate_splits(total_length: int, ratios: tuple = (0.7, 0.1, 0.2)) -> tuple:
    if total_length < 0:
        raise ValueError("Dataset length cannot be negative")
        
    if sum(ratios) != 1:
        raise ValueError("Split ratios must sum to 1")
    
    # Calculate sizes and handle rounding
    train_size = int(total_length * ratios[0])
    val_size = int(total_length * ratios[1])
    # Assign remaining samples to test to ensure total adds up
    test_size = total_length - train_size - val_size
    
    return (train_size, val_size, test_size)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        scheduler.step(result['val_loss'])
    return history

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

def main():
    data_dir = os.getenv('DATA_PATH')
    classes = os.listdir(data_dir)
    print(classes)
    
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])
    
    dataset = ImageFolder(data_dir, transform=transformations)
    
    random_seed = 42
    torch.manual_seed(random_seed)
    
    print(len(dataset))
    train_len, val_len, test_len = calculate_splits(len(dataset))
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    
    class ResNet(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            # Use a pretrained model
            self.network = models.resnet50(pretrained=True)
            # Replace last layer
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, xb):
            out = self.network(xb)
            return torch.sigmoid(self.dropout(out))
    
    def predict_image(img, model):
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        prob, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return dataset.classes[preds[0].item()]

    def predict_external_image(image_name):
        image = Image.open(Path('./' + image_name))
        example_image = transformations(image)
        plt.imshow(example_image.permute(1, 2, 0))
        print("The image resembles", predict_image(example_image, model) + ".")

    model = ResNet()
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)
    model = to_device(ResNet(), device)
    evaluate(model, val_dl)
    num_epochs = 8
    opt_func = torch.optim.Adam
    lr = 5.5e-5

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_accuracies(history)

    predict_external_image("cans.jpg")
    predict_external_image("plastic.jpg")
    predict_external_image("image-1177993601.jpg")
    predict_external_image("container.jpg")
    predict_external_image("storage_bins.jpg")
    predict_external_image("garbage_bin.jpg")

    torch.save(model.state_dict(), "classifier_with_bin.pth")

if __name__ == '__main__':
    freeze_support()
    main()