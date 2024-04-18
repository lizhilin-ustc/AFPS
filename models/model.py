import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math

#### 

class Model(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments, pooling_type=""):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.num_segments = num_segments

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0)
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0)
        )

        self.discrim = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=512, out_channels=1 , kernel_size=1, padding=0)
        )

        self.action_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.action_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )


    def forward(self, x, config):
        input = x.permute(0, 2, 1)  # x.shape=[16,750,2048]     input.shape=[16,2048,750]

        flow = input[:, 1024:, :]
        rgb = input[:, :1024, :]

        action_flow = torch.sigmoid(self.action_flow(flow))
        action_rgb = torch.sigmoid(self.action_rgb(rgb))

        # import pdb; pdb.set_trace()

        emb = self.base_module(input)   # [16, 512, 750]
        discrim = torch.sigmoid(self.discrim(emb))       # [16, 1, 750]

        act =  config.hp_alpha * action_rgb + (1-config.hp_alpha) * action_flow
        non_act = 1 - act

        cas = self.cls(emb).permute(0, 2, 1)
        cas_fg = cas * discrim.permute(0, 2, 1)
        cas_fp = cas * non_act.permute(0, 2, 1)

        return cas, cas_fg, cas_fp, discrim, (action_flow, action_rgb)
