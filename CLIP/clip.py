import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


class ImageTextDataset(Dataset):
    def __init__(self, image_paths, captions, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        if self.transform:
            image =  self.transform(image)
        text = self.captions[item]

        return image, text


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)

    def forward(self, images):
        return self.resnet(images)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=77):
        super(TextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.transformer = nn.Transformer(embed_size, nhead=8, num_encoder_layers=6)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, tokens):
        positions = torch.arange(0, tokens.size(1)).unsqueeze(0).to(tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.transformer(x, x)
        return self.fc(x.mean(dim=1))


class CLIP(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder(image_embed_dim)
        self.text_encoder = TextEncoder(vocab_size=30522, embed_dim=text_embed_dim)

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        return image_features, text_features


def contrastive_loss(image_features, text_features, temperature=0.1):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)

    return loss

def train_clip(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, texts in dataloader:
        images, texts = images.to(device), texts.to(device)

        optimizer.zero_grad()
        image_features, text_features = model(images, texts)
        loss = contrastive_loss(image_features, text_features)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    embed_dim = 512
    batch_size = 16
    learning_rate = 3e-4
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageTextDataset() # Image + Caption으로 구성된 데이터셋 필요
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CLIP(embed_dim, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(10)):
        loss = train_clip(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")