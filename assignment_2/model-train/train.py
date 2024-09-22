import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from model.model import Net
from util.utils import get_args  # Import the argument parser function

checkpoint_path = '/opt/mount/model/mnist_cnn.pt'

def train(model, device, train_loader, optimizer, epoch, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def main():
    args = get_args()  # Get the arguments
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('/opt/mount/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    # Model and Optimizer
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        print(f'Resuming from checkpoint: Epoch {epoch}')
    else:
        epoch = 1
    
    # Train for 1 epoch
    train(model, device, train_loader, optimizer, epoch, args)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print("model save success")
    print(checkpoint_path)

if __name__ == '__main__':
    main()
