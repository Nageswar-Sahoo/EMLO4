import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from model.model import Net
from util.utils import get_args  # Import the argument parser function

checkpoint_path = '/opt/mount/model/mnist_cnn.pt'
metrics_path = '/opt/mount/model/eval_results.json'

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def main():
    args = get_args()  # Get the arguments
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("model loaded")

    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST('/opt/mount/data', train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
    print("model loaded 1")

    # Model
    model = Net().to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No checkpoint found.')
        return
    
    # Evaluate model
    test_loss, accuracy = evaluate(model, device, test_loader)
    
    # Save evaluation metrics
    metrics = {'test_loss': test_loss, 'accuracy': accuracy}
    print(metrics)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        

if __name__ == '__main__':
    main()
