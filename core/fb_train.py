import os
import torch
import torch.nn as nn
import numpy as np
import random
import tqdm
import argparse

from resnet18 import ResNet18
from model import MultimodalModel
from follow_model import FollowerModel
from fb_dataset import FacebookDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights

parser = argparse.ArgumentParser()

parser.add_argument('--mode', nargs='+', default=["time", "image", "text"])
parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction)
parser.add_argument('--freeze', action=argparse.BooleanOptionalAction)
parser.add_argument('--composite', action=argparse.BooleanOptionalAction)
parser.add_argument('--label', type=str, default="view")

args = parser.parse_known_args()[0]

if args.pretrained:
    pretrained = 'pretrained'
else:
    pretrained = 'no-pretrained'

if args.freeze:
    freeze = 'frozen'
else:
    freeze = 'no-frozen'

if args.composite:
    composite = 'composite'
else:
    composite = 'no-composite'

if args.label == 'rate':
    task_type = 'regression'
if args.label == 'sentiment':
    task_type = 'classification'
if args.label == 'reaction':
    task_type = 'regression'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(pretrained)
print(freeze)
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

# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128)), transforms.Normalize([0.454, 0.423, 0.399], [0.301, 0.292, 0.298])])
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize([0.454, 0.423, 0.399], [0.301, 0.292, 0.298])])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize([0.5], [0.5])])
# dataset = FacebookDataset(root_dir = "./", img_dir = 'fb_img', data_file = 'facebook_data.csv', text_file = 'fb_text_segmented.txt', label_type = args.label, transform  = transform)
dataset = FacebookDataset(root_dir = "./", label_type = args.label, transform  = transform)
if args.composite:
    dataset = FacebookDataset(root_dir = "./", img_dir = "composite_images_", label_type = args.label, transform  = transform)

rng = torch.Generator().manual_seed(seed)
### Train-test split

np.random.seed(seed)
post_ids = np.random.permutation(list(set(dataset.ids_to_postids.values())))
data_len = len(post_ids)
train_size = int(0.8 * data_len)
val_size = int(0.1 * data_len)
test_size = data_len - train_size - val_size
train_ids = [idx for idx in range(len(dataset)) if dataset.ids_to_postids[idx] in post_ids[:train_size]]
val_ids = [idx for idx in range(len(dataset)) if dataset.ids_to_postids[idx] in post_ids[train_size: train_size + val_size]]
test_ids = [idx for idx in range(len(dataset)) if dataset.ids_to_postids[idx] in post_ids[train_size + val_size:]]


print("Train size: ", len(train_ids))
print("Validation size: ", len(val_ids))
print("Test size: ", len(test_ids))
train_dataset = Subset(dataset, train_ids)
val_dataset = Subset(dataset, val_ids)
test_dataset = Subset(dataset, test_ids)
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = rng)

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size = batch_size, num_workers=32, shuffle = True, generator = rng)
val_dl = DataLoader(val_dataset, num_workers=32, batch_size = batch_size * 2)
test_dl = DataLoader(test_dataset, num_workers=32, batch_size = batch_size * 2)

if task_type == 'regression':
    # model = ResNet18(image_channels = 3, num_classes = 1).to(device)
    # # New weights with accuracy 80.858%
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    model = MultimodalModel(time_d = 5, seq_len = 128, mode = args.mode).to(device)
    if args.pretrained:
        model = torch.load(os.path.join(dataset.root_dir, 'model/best_model.pt'), map_location = device)
        if args.freeze:
            for n, c in list(model.resnet18.named_children())[:7]:
                for param in c.parameters():
                    param.requires_grad = False
            for n, c in list(model.bert_model.named_children()):
                for param in c.parameters():
                    param.requires_grad = False
    model.regression_head = nn.Identity()
    follow_model = FollowerModel().to(device)
    model = model.to(device)
    loss_fn = nn.MSELoss()

if task_type == 'classification':
    num_classes = int(max(dataset.labels) + 1)
    model = MultimodalModel(time_d = 5, seq_len = 128, mode = args.mode)
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # model.fc = nn.Linear(2048, num_classes)
    # for n, c in list(model.named_children())[:7]:
    #     for param in c.parameters():
    #         param.requires_grad = False
    model = model.to(device)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

model.float()
epoch_num = 50
optim = torch.optim.AdamW(model.parameters(),
                  lr = 3e-5,
                  eps = 1e-8
                )

# acc_fn = lambda logit, label: torch.mean((torch.argmax(logit, dim = -1) == label).float())
acc_fn = lambda logit, label: torch.mean(((torch.sigmoid(logit) > 0.5) == label).float())


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
            img, label, time_feats, text, follow_count = batch
            img = img.to(torch.float32)
            img = img.to(device)
            time_feats = time_feats.to(device)
            text = text.to(device)
            follow_count = follow_count.view(-1, 1).to(dtype = torch.float32, device = device)
            label = torch.tensor(label, dtype = torch.float32, device = device)
            label = label.view(-1, 1)
            logits = model(img, time_feats, text)
            logits = follow_model(logits, follow_count)
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
                img, label, time_feats, text, follow_count = batch
                img = img.to(device)
                time_feats = time_feats.to(device)
                text = text.to(device)
                follow_count = follow_count.view(-1, 1).to(dtype = torch.float32, device = device)
                label = torch.tensor(label, dtype = torch.float32, device = device)
                label = label.view(-1, 1)
                logits = model(img, time_feats, text)
                logits = follow_model(logits, follow_count)
                val_loss = loss_fn(logits, label)
                val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {e + 1}: Training Loss {avg_train_loss:.4f} Validation Loss {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, os.path.join(dataset.root_dir, f'model/fb_test/best_model_{pretrained}_{freeze}_{composite}.pt'))
            torch.save(follow_model, os.path.join(dataset.root_dir, f'model/fb_test/best_followmodel_{pretrained}_{freeze}_{composite}.pt'))

if task_type == 'classification':

    seed_everything(seed)

    losses = []
    accs = []
    optim.zero_grad()
    best_val_loss = float('inf')
    for e in range(epoch_num):
        losses = []
        accs = []
        for batch in tqdm.tqdm(iter(train_dl)):
            img, label, time_feats, text, _ = batch
            img = img.to(device)
            time_feats = time_feats.to(device)
            text = text.to(device)
            img = img.to(torch.float32)
            label = torch.tensor(label).type(torch.float32).view(-1, 1)
            label = label.to(device)
            logits = model(img, time_feats, text)
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
            for batch in tqdm.tqdm(iter(val_dl)):
                img, label, time_feats, text, _ = batch
                img = img.to(device)
                img = img.to(torch.float32)
                time_feats = time_feats.to(device)
                text = text.to(device)
                label = torch.tensor(label).type(torch.float32).view(-1, 1)
                label = label.to(device)
                logits = model(img, time_feats, text)
                val_acc = acc_fn(logits, label)
                val_loss = loss_fn(logits, label)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = sum(val_accs) / len(val_accs)
            print(f'Epoch {e + 1}: Training Loss {avg_train_loss:.4f} Training Accuracy {avg_train_acc * 100:.2f}% - Validation Loss {avg_val_loss:.4f} Validation Accuracy {avg_val_acc * 100:.2f}%')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, os.path.join(dataset.root_dir, f'model/category_model/best_model.pt'))