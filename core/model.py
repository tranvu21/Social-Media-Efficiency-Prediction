import torch
import torch.nn as nn

from resnet18 import ResNet18
from transformers import AutoModel, AutoTokenizer

class MultimodalModel(nn.Module):
    def __init__(self, time_d = 5, seq_len = 64, mode = ["all"]):
        super().__init__()
        self.mode = mode ##Modality
        if mode == ["all"]:
            self.mode = ["image", "text", "time"]
        self.seq_len = seq_len
        self.resnet18 = ResNet18(image_channels = 3, num_classes = 16)
        self.resnet18.float()
        self.time_model = nn.Sequential(
            nn.Linear(time_d, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.bert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")

        self.bert_head_1 = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bert_head_2 =  nn.Sequential(
            nn.Linear(self.seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(16 * len(self.mode), 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, img, time_feats, text):
        b, _, _, _ = img.size()

        x = torch.empty(b, 0).cuda()

        if "image" in self.mode:
            img = self.resnet18(img)
            x = torch.cat((x, img), dim = 1)
            
        if "time" in self.mode:
            time_feats = self.time_model(time_feats)
            x = torch.cat((x, time_feats), dim = 1)

        if "text" in self.mode:
            text = text.to(dtype = torch.int32)
        
            text_feats = self.bert_model(text).last_hidden_state
            text_feats = self.bert_head_1(text_feats)
            text_feats = text_feats.view(b, self.seq_len)
            text_feats = self.bert_head_2(text_feats)
            x = torch.cat((x, text_feats), dim = 1)

        x = self.regression_head(x)
        return x


