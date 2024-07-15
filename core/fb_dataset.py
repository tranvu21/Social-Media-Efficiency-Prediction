import ast
import os
import numpy as np
import json
import time
import torch
import pandas as pd
import regex as re
import tqdm

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

class FacebookDataset(Dataset):

    def __init__(self, root_dir, img_dir = 'fb_img_2', data_file = 'facebook_data_v2.csv', text_file = 'fb_text_segmented_2.txt', label_type = 'reaction', follower_count = True, transform=None):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.data_file = data_file
        self.text_file = text_file
        self.label_type = label_type
        self.transform = transform
        self.data = pd.read_csv(os.path.join(root_dir, data_file))

        if self.label_type == "rate":
            self.full_labels = list(self.data['reactions'] / self.data['page_follow'])
            self.labels = self.normalize(self.full_labels)
            self.label_mean = np.mean(np.array(self.full_labels))
            self.label_std = np.std(np.array(self.full_labels))
        elif self.label_type == "sentiment":
            self.full_labels = np.array(list(self.data['average_score']))
            self.labels = np.where(self.full_labels > 0, 1, 0)

            # self.full_labels = list(self.data['average_score'])
            # self.labels = self.normalize(self.full_labels)
            # self.label_mean = np.mean(np.array(self.full_labels))
            # self.label_std = np.std(np.array(self.full_labels))
        elif self.label_type == "reaction":
            self.full_labels = list(self.data['reactions'])
            self.labels = self.normalize(self.full_labels)
            self.label_mean = np.mean(np.array(self.full_labels))
            self.label_std = np.std(np.array(self.full_labels))

        self.page_follow = self.normalize(list(self.data['page_follow']))
        self.post_ids = list(self.data['post_id'])

        def num_sort(test_string):
            return list(map(int, re.findall(r'\d+', test_string)))[0]

        self.img_files = sorted(os.listdir(os.path.join(self.root_dir, self.img_dir)), key = num_sort)
        self.img_files = [x[:-4] for x in self.img_files]
        self.ids_to_dataids = {}
        self.postids_to_dataids = {}
        self.ids_to_postids = {}
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.tokd_text = self.get_tokd_text()

        for idx, post_id in enumerate(self.post_ids):
            self.postids_to_dataids[post_id] = idx

        for idx, img_id in enumerate(self.img_files):
            self.ids_to_dataids[idx] = self.postids_to_dataids[int(img_id.split("-")[0])]
            self.ids_to_postids[idx] = int(img_id.split("-")[0])

        self.time_feats = self.get_time_features()
            

    def normalize(self, arr:list) -> np.array:
        arr = np.array(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / std

    def get_text(self):
        return [x['caption'] for x in self.data]
    
    def get_time_features(self):
        postdates = list(self.data['publish_date'])
        posthours = list(self.data['publish_time'])
        date_data = []
        hour_data = []

        for date in postdates:
            if "-" in date:
                date_data.append(time.strptime(date, '%Y-%m-%d'))
            elif "/" in date:
                date_data.append(time.strptime(date, '%m/%d/%Y'))
            elif date == "Unknown":
                date_data.append(time.strptime("01/01/2024", '%m/%d/%Y'))

        for hour in posthours:
            if hour == "Unknown":
                hour_data.append(time.strptime("19:00:00", '%H:%M:%S'))
            else:
                hour_data.append(time.strptime(hour, '%H:%M:%S'))
        
        frac_years = self.normalize([t[0] + (t[7] / 365.0) for t in date_data])
        dec_years = self.normalize([np.sin(2 * np.pi * t[0]/2024) for t in date_data])
        cyclic_date = [np.sin(2 * np.pi * t[7]/365.0) for t in date_data]
        cyclic_wday = [np.sin(2 * np.pi * t[6]/7.0) for t in date_data]
        cyclic_hours = [np.sin(2 * np.pi * t[3]/24.0) for t in hour_data]

        return np.array([frac_years, dec_years, cyclic_date, cyclic_wday, cyclic_hours])
    
    def get_tokd_text(self):
        with open(os.path.join(self.root_dir, self.text_file), "r") as f:
            segmented_text = f.readlines()
        
        print("Load tokenized text....")
        tokd_inputs = torch.empty(0, 128)
        for i in tqdm.tqdm(range(0, len(segmented_text), 512)):
            inp = self.tokenizer(segmented_text[i: i+512], return_tensors = 'pt', max_length = 128, padding = 'max_length', truncation = True)
            tokd_inputs = torch.vstack((tokd_inputs, inp.input_ids))

        return tokd_inputs

    def read_img(self, idx):
        img_id = self.img_files[idx]
        img_path = os.path.join(os.path.join(self.root_dir, self.img_dir), f'{img_id}.jpg')
        image = io.imread(img_path, as_gray = False)
        if image.shape[-1] != 3:
            x = np.ones((image.shape[0], image.shape[1], 3))
            x[:, :, 0] = image
            x[:, :, 1] = image
            x[:, :, 2] = image
            image = x
        return image, img_id

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        image, img_id = self.read_img(idx)
        if image.shape[0] == 1:
            print(img_id)
        data_id = self.ids_to_dataids[idx]

        if self.transform:
            image = self.transform(image)
        label = self.labels[data_id]
        time_feats = torch.tensor(self.time_feats[:, data_id], dtype = torch.float32)
        text = self.tokd_text[data_id]
        follow_count = self.page_follow[data_id]

        return image, label, time_feats, text, follow_count


if __name__ == "__main__":
    d = FacebookDataset(root_dir = "./")
    print(next(iter(d)))