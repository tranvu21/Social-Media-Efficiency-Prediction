import os
import torch
import numpy as np
import random
import tqdm
import pandas as pd
import argparse

from follow_model import FollowerModel
from scipy import stats
from fb_dataset import FacebookDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction)
parser.add_argument('--freeze', action=argparse.BooleanOptionalAction)
parser.add_argument('--composite', action=argparse.BooleanOptionalAction)
parser.add_argument('--label', type=str, default="view")

args = parser.parse_known_args()[0]

if args.label == "sentiment":
    task_type = "classification"
else:
    task_type = "regression"

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
print(pretrained)
print(freeze)
print(composite)

seed = 42
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize([0.454, 0.423, 0.399], [0.301, 0.292, 0.298])])
if args.composite:
    dataset = FacebookDataset(root_dir = "./", img_dir = "composite_images_", label_type = args.label, transform  = transform)
else:
    dataset = FacebookDataset(root_dir = "./", label_type = args.label, transform  = transform)
# dataset = FacebookDataset(root_dir = "./", img_dir = 'fb_img', data_file = 'facebook_data.csv', text_file = 'fb_text_segmented.txt', transform  = transform)

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

print("Test size: ", len(test_ids))
train_dataset = Subset(dataset, train_ids)
val_dataset = Subset(dataset, val_ids)
test_dataset = Subset(dataset, test_ids)
batch_size = 16
train_dl = DataLoader(train_dataset, batch_size = batch_size, num_workers=32, shuffle = True, generator = rng)
val_dl = DataLoader(val_dataset, num_workers=32, batch_size = batch_size * 2)
test_dl = DataLoader(test_dataset, num_workers=32, batch_size = batch_size * 2)

if task_type == "regression":
    model = torch.load(os.path.join(dataset.root_dir, f'model/fb_test/best_model_{pretrained}_{freeze}_{composite}.pt'), map_location = device)
    follow_model = torch.load(os.path.join(dataset.root_dir, f'model/fb_test/best_followmodel_{pretrained}_{freeze}_{composite}.pt'), map_location = device)
elif task_type == "classification":
    model = torch.load(os.path.join(dataset.root_dir, f'model/category_model/best_model.pt'), map_location = device)

preds = torch.empty((0,1), device = device)
labels = torch.empty((0,1), device = device)
model.eval()

if task_type == "regression":
    with torch.no_grad():
        for batch in tqdm.tqdm(iter(test_dl)):
            img, label, time_feats, text, follow_count = batch
            img = img.to(device)
            text = text.to(device)
            time_feats = time_feats.to(device)
            follow_count = follow_count.view(-1, 1).to(dtype = torch.float32, device = device)
            label = torch.tensor(label, dtype = torch.float32, device = device).view(-1, 1)
            logits = model(img, time_feats, text)
            logits = follow_model(logits, follow_count)
            # logits = logits * dataset.label_std + dataset.label_mean
            # label = label * dataset.label_std + dataset.label_mean
            preds = torch.vstack((preds, logits))
            labels = torch.vstack((labels, label))

    print(f'Correlation: {stats.spearmanr(preds.cpu().numpy().reshape(-1), labels.cpu().numpy().reshape(-1)).statistic}')
    print(f'MAE: {torch.mean(torch.abs(preds - labels))}')
    print(f'MSE: {torch.mean(torch.square(preds - labels))}')

elif task_type == "classification":
    acc_fn = lambda logit, label: torch.mean(((torch.sigmoid(logit) > 0.5) == label).float())
    with torch.no_grad():
        for batch in tqdm.tqdm(iter(test_dl)):
            img, label, time_feats, text, _ = batch
            img = img.to(device)
            text = text.to(device)
            time_feats = time_feats.to(device)
            label = torch.tensor(label, dtype = torch.float32, device = device).view(-1, 1)
            logits = model(img, time_feats, text).view(-1, 1)
            # logits = logits * dataset.label_std + dataset.label_mean
            # label = label * dataset.label_std + dataset.label_mean
            preds = torch.vstack((preds, logits))
            labels = torch.vstack((labels, label))
    print(f'Accuracy: {acc_fn(preds, labels):.5f}')
