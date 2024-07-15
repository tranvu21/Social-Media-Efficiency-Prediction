import ast
import os
import numpy as np
import json
import time
import torch
import tqdm

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

class FlickrDataset(Dataset):

    def __init__(self, root_dir, img_dir, id_file, label_file, label_type = 'view', transform=None):
        self.id_file = id_file
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_file = label_file
        self.label_type = label_type
        self.transform = transform
        with open(os.path.join(root_dir, id_file)) as f:
            arr = f.read()
        f.close()
        self.all_ids = ast.literal_eval(arr)
        self.ids_to_imgids = {}

        # self.bert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        self.labels, self.label_mean, self.label_std = self.get_labels()

        for i, id_ in enumerate(self.all_ids):
            self.ids_to_imgids[i] = id_

    def get_labels(self):
        if self.label_type == 'view':
            with open(os.path.join(self.root_dir, self.label_file)) as f:
                labels = [float(y[:-1]) for y in f.readlines()[1:]]
            f.close()

            label_mean = np.mean(labels)
            label_std = np.std(labels)
            labels = (labels - label_mean) / label_std

            return labels, label_mean, label_std
        
        if self.label_type == 'category':
            f = open(os.path.join(self.root_dir, 'train_category.json'))
            data = json.load(f)
            labels = [x['Category'] for x in data]
            classes = sorted(set(labels))
            class_to_id = dict(zip(classes, range(len(classes))))
            labels = [class_to_id[label] for label in labels]

            return labels, None, None

        raise Exception("Label type does not match!")


    def read_img(self, idx):
        img_id = self.ids_to_imgids[idx]
        img_path = os.path.join(os.path.join(self.root_dir, self.img_dir), f'{img_id}.jpg')
        image = io.imread(img_path)
        return image, img_id

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):

        image, img_id = self.read_img(idx)

        if self.transform:
            image = self.transform(image)
        label = self.labels[img_id]

        return image, label


class CombinedDataset(FlickrDataset):
    def __init__(self, root_dir, img_dir, id_file, label_file, label_type = 'view', transform=None, time_file = "train_temporalspatial.json"):
        super().__init__(root_dir, img_dir, id_file, label_file, label_type, transform)
        self.time_file = time_file
        self.time_feats = self.get_time_features()
        self.tokd_text = self.get_tokd_text()

    def get_time_features(self):
        f = open(os.path.join(self.root_dir, self.time_file))
        data = json.load(f)
        postdates = [x['Postdate'] for x in data]
        postdates = list(map(int, postdates))
        times = [time.localtime(x) for x in postdates]
        frac_years = self.normalize([t[0] + (t[7] / 365.0) for t in times])
        dec_years = [np.sin(2 * np.pi * t[0]/2020) for t in times]
        cyclic_date = [np.sin(2 * np.pi * t[7]/365.0) for t in times]
        cyclic_wday = [np.sin(2 * np.pi * t[6]/7.0) for t in times]
        cyclic_hours = [np.sin(2 * np.pi * t[3]/24.0) for t in times]

        return np.array([frac_years, dec_years, cyclic_date, cyclic_wday, cyclic_hours])

    def normalize(self, arr:list) -> np.array:
        arr = np.array(arr)
        mean = np.mean(arr)
        std = np.std(arr)

        return (arr - mean) / std

    def get_tokd_text(self):
        with open('vie_text_segmented.txt', 'r') as f:
            segmented_text = f.readlines()
        
        print("Load tokenized text....")
        tokd_inputs = torch.empty(0, 128)
        for i in tqdm.tqdm(range(0, len(segmented_text), 512)):
            inp = self.tokenizer(segmented_text[i: i+512], return_tensors = 'pt', max_length = 128, padding = 'max_length', truncation = True)
            tokd_inputs = torch.vstack((tokd_inputs, inp.input_ids))

        return tokd_inputs


    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):

        image, label = super().__getitem__(idx = idx)
        real_idx = self.ids_to_imgids[idx]
        time_feats = torch.tensor(self.time_feats[:, real_idx], dtype = torch.float32)
        tokd_text = self.tokd_text[real_idx]

        return image, label, time_feats, tokd_text

if __name__ == "__main__":
    dataset = CombinedDataset(root_dir = "./", img_dir = 'img', id_file = 'all_ids.txt', label_file = 'train_label.txt', label_type = "category", \
                            transform  = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize([0.454, 0.423, 0.399], [0.301, 0.292, 0.298])]))
    
    print(next(iter(dataset))[2])