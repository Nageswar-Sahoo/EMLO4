import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from model.model import Net
from util.utils import get_args  # Import the argument parser function

checkpoint_path = '/opt/mount/model/mnist_cnn.pt'
results_folder = '/opt/mount/results/'

def infer(model, device, test_loader):
    model.eval()
    os.makedirs(results_folder, exist_ok=True)
    
    with torch.no_grad():
     saved_preds = set()  # To keep track of saved predictions
     for i, (data, _) in enumerate(test_loader):
        if i >=5:  # Only infer 5 images
            break
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        
        for idx in range(len(data)):
            current_pred = pred[idx].item()

            # Only save the image if the prediction hasn't been saved before
            if current_pred not in saved_preds:

                # Convert and save image
                img = data[idx].cpu().numpy().squeeze() * 255
                img = Image.fromarray(img).convert("L")
                img.save(os.path.join(results_folder, f'{i}.png'))

def main():
    args = get_args()  # Get the arguments
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST('/opt/mount/data', train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    # Model
    model = Net().to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No checkpoint found.')
        return
    
    # Run inference on 5 random images
    infer(model, device, test_loader)
    print("infer completed")

if __name__ == '__main__':
    main()
