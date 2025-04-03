import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from breed_info import get_all_breeds

# Custom dataset class
class DogBreedDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all unique breeds and sort them
        all_breeds = sorted(self.labels_df['breed'].unique())
        # Create mapping from breed to index
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(all_breeds)}
        print(f"Found {len(all_breeds)} unique breeds")
        print("All breeds in dataset:")
        for breed in all_breeds:
            print(f"- {breed}")
        print("\nFirst few breed mappings:")
        print(dict(list(self.breed_to_idx.items())[:5]))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.labels_df.iloc[idx]['id']}.jpg")
        image = Image.open(img_name).convert('RGB')
        breed = self.labels_df.iloc[idx]['breed']
        label = self.breed_to_idx[breed]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get number of breeds from breed_info
num_breeds = len(get_all_breeds())
print(f"Number of breeds in breed_info.py: {num_breeds}")

# Load dataset
dataset = DogBreedDataset(
    csv_file='datasets/labels.csv',
    root_dir='datasets/train',
    transform=transform
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.breed_to_idx))  # Use actual number of breeds from dataset

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train():
    print(f">>> Starting training for {len(dataset.breed_to_idx)} breeds...")
    for epoch in range(2):  # Adjust epochs as needed
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, Average Loss: {epoch_loss:.4f}")
    
    # Save model
    model_path = Path('models/breed_model/mobilenetv2_dogbreeds.pth')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f">>> Model saved to {model_path}")

if __name__ == "__main__":
    train()