import os
import torch
import torch.nn as nn
import numpy as np
import random
import tqdm

from resnet18 import ResNet18
from model import MultimodalModel
from img_dataset import FlickrDataset, CombinedDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights


label_type = 'view'

if label_type == 'view':
    task_type = 'regression'
if label_type == 'category':
    task_type = 'classification'

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

dataset = CombinedDataset(root_dir = "./", img_dir = 'img', id_file = 'all_ids.txt', label_file = 'train_label.txt', label_type = label_type, \
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

if task_type == 'regression':
    # model = ResNet18(image_channels = 3, num_classes = 1).to(device)
    # # New weights with accuracy 80.858%
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    model = MultimodalModel(time_d = 5, seq_len = 128).to(device)
    loss_fn = nn.MSELoss()

if task_type == 'classification':
    num_classes = max(dataset.labels) + 1
    # model = ResNet18(image_channels = 3, num_classes = num_classes).to(device)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)
    for n, c in list(model.named_children())[:7]:
        for param in c.parameters():
            param.requires_grad = False
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

model.float()
epoch_num = 30
optim = torch.optim.AdamW(model.parameters(),
                  lr = 3e-3,
                  eps = 1e-8
                )

acc_fn = lambda logit, label: torch.mean((torch.argmax(logit, dim = -1) == label).float())


if task_type == 'regression':

    seed_everything(seed)

    losses = []
    optim.zero_grad()
    best_val_loss = float('inf')
    for e in range(epoch_num):
        model.train()
        losses = []
        accs = []
        for batch in tqdm.tqdm(iter(train_dl)):
            img, label, time_feats, text = batch
            img = img.to(device)
            time_feats = time_feats.to(device)
            text = text.to(device)
            label = torch.tensor(label, dtype = torch.float32, device = device)
            label = label.view(-1, 1)
            logits = model(img, time_feats, text)
            loss = loss_fn(logits, label)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            losses.append(loss.item())

            loss.backward()
            optim.step()
            optim.zero_grad()
            
        avg_train_loss = sum(losses) / len(losses)

        model.eval()
        val_accs = []
        val_losses = []
        with torch.no_grad():
            for batch in tqdm.tqdm(iter(val_dl)):
                img, label, time_feats, text = batch
                img = img.to(device)
                time_feats = time_feats.to(device)
                text = text.to(device)
                label = torch.tensor(label, dtype = torch.float32, device = device)
                label = label.view(-1, 1)
                logits = model(img, time_feats, text)
                val_loss = loss_fn(logits, label)
                val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {e + 1}: Training Loss {avg_train_loss:.4f} Validation Loss {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, os.path.join(dataset.root_dir, 'model/best_model_text_time.pt'))

if task_type == 'classification':

    seed_everything(seed)

    losses = []
    accs = []
    optim.zero_grad()
    best_val_loss = float('inf')
    for e in tqdm.tqdm(range(epoch_num)):
        model.train()
        losses = []
        accs = []
        for batch in iter(train_dl):
            img, label = batch
            img = img.to(device)
            label = torch.tensor(label).type(torch.LongTensor).view(-1)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            losses.append(loss.item())
            accs.append(acc)

            loss.backward()
            optim.step()
            optim.zero_grad()
            
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(accs) / len(accs)

        model.eval()
        val_accs = []
        val_losses = []
        with torch.no_grad():
            for batch in iter(val_dl):
                img, label = batch
                img = img.to(device)
                label = torch.tensor(label).type(torch.LongTensor).view(-1)
                label = label.to(device)
                logits = model(img)
                val_acc = acc_fn(logits, label)
                val_loss = loss_fn(logits, label)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = sum(val_accs) / len(val_accs)
            print(f'Epoch {e + 1}: Training Loss {avg_train_loss:.4f} Training Accuracy {avg_train_acc * 100:.2f}% - Validation Loss {avg_val_loss:.4f} Validation Accuracy {avg_val_acc * 100:.2f}%')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, 'SMP/model/category_model/best-model.pt')