import os
import torch
import numpy as np
import random
import tqdm
import pandas as pd
from scipy import stats

from img_dataset import FlickrDataset, CombinedDataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

seed = 42
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

dataset = CombinedDataset(root_dir = "./", img_dir = 'img', id_file = 'all_ids.txt', label_file = 'train_label.txt', label_type = 'view', \
                        transform  = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize([0.454, 0.423, 0.399], [0.301, 0.292, 0.298])]))

rng = torch.Generator().manual_seed(seed)
### Train-test split
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = rng)

batch_size = 128
train_dl = DataLoader(train_dataset, batch_size = batch_size, num_workers=32, shuffle = True, generator = rng)
val_dl = DataLoader(val_dataset, num_workers=32, batch_size = batch_size * 2)
test_dl = DataLoader(test_dataset, num_workers=32, batch_size = batch_size * 2)

model = torch.load(os.path.join(dataset.root_dir, 'model/best_model.pt'), map_location = device)

preds = torch.empty((0,1), device = device)
labels = torch.empty((0,1), device = device)
model.eval()
with torch.no_grad():
    for batch in tqdm.tqdm(iter(test_dl)):
        img, label, time_feats = batch
        img = img.to(device)
        time_feats = time_feats.to(device)
        label = torch.tensor(label, dtype = torch.float32, device = device).view(-1, 1)
        logits = model(img, time_feats)
        logits = logits * dataset.label_std + dataset.label_mean
        label = label * dataset.label_std + dataset.label_mean
        preds = torch.vstack((preds, logits))
        labels = torch.vstack((labels, label))

print(f'Correlation: {stats.spearmanr(preds.cpu().numpy().reshape(-1), labels.cpu().numpy().reshape(-1)).statistic}')
print(f'MAE: {torch.mean(torch.abs(preds - labels))}')
print(f'MSE: {torch.mean(torch.square(preds - labels))}')