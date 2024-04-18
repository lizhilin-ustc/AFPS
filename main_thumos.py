
# 48.5
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
# from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature

from models.model import Model
from utils.loss import CrossEntropyLoss, GeneralizedCE, GeneralizedCE2, BCE, BCE1, Focal
from config.config_thumos import Config, parse_args, class_dict

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        # self.net = ModelFactory.get_model(config.model_name, config)
        self.net = Model(config.len_feature, config.num_classes, config.num_segments)
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)   # 0.0005
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)
        self.Lgce2 = GeneralizedCE2(q=self.config.q_val)
        self.bce = BCE()
        self.bce1 = BCE1()
        self.focal = Focal(gamma=2)

        # parameters
        self.best_mAP = -1 # init
        self.step = 0
        self.total_loss_per_epoch = 0
        self.hp_lambda = config.hp_lambda


    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments

    def calculate_pesudo_target1(self, batch_size, topk_indices):
        cls_agnostic_gt = []
        for b in range(batch_size):
            topk_indices_b = topk_indices[b, :]          # [75, 1]
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            cls_agnostic_gt_b[0, 0, topk_indices_b[:, 0]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)         # B, 1, num_segments
        
    def calculate_all_losses(self, epoch, cas_top, cas_top_fg, cas_top_fp, _label, action_flow, action_rgb, discrim, cas_top_ca):

        base_loss = self.criterion(cas_top, _label)
        fg_loss = self.criterion(cas_top_fg, _label)
        label_bg = torch.ones_like(_label).cuda()
        fp_loss = self.criterion(cas_top_fp, label_bg)
        label_ca = torch.ones(cas_top_ca.shape[0], 2).cuda() 
        loss_fn = nn.BCELoss()
        sta_loss = loss_fn(cas_top_ca, label_ca) + F.mse_loss(action_rgb, action_flow)

        cost =  base_loss + self.hp_lambda*sta_loss + fg_loss + fp_loss

        return cost

    
    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap

                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))
                # torch.save(self.best_mAP, os.path.join(self.config.model_path, "best_mAP.txt"))
                with open(os.path.join(self.config.model_path, "best_mAP.txt"), 'a') as file:
                    write_str = '%f\n' % (self.best_mAP)
                    file.write(write_str)

            print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f} mean_ap={:5.2f} best_map={:5.2f}".format(
                    epoch, self.step, self.total_loss_per_epoch, test_acc * 100, mean_ap*100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0


    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        # l = []
        for epoch in range(self.config.num_epochs):
            
            for _data, _label, temp_anno, vid_name, vid_num_seg in self.train_loader:

                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()
                # forward pass
                cas, cas_fg, cas_fp, discrim, (action_flow, action_rgb) = self.net(_data, self.config)

                combined_cas = misc_utils.instance_selection_function(torch.softmax(cas_fg, -1), action_flow.permute(0, 2, 1), action_rgb.permute(0, 2, 1))   # *****
                _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 12, dim=1)  # combined_cas  12
                cas_top_fg = torch.mean(torch.gather(cas_fg, 1, topk_indices), dim=1)

                _, topk_indices1 = torch.topk(torch.softmax(cas, -1), self.config.num_segments // 8, dim=1)  # //8  torch.softmax(cas, -1)
                cas_top = torch.mean(torch.gather(cas, 1, topk_indices1), dim=1)  

                _, topk_indices_fp = torch.topk(torch.softmax(cas_fp, -1), self.config.num_segments // 8, dim=1)  # 8
                cas_top_fp = torch.mean(torch.gather(cas_fp, 1, topk_indices_fp), dim=1)  # cas_fp
                
                action = (1-self.config.hp_alpha) * action_flow.permute(0, 2, 1) + self.config.hp_alpha * action_rgb.permute(0, 2, 1)   

                K1, K2 = self.config.num_segments//10, self.config.num_segments - self.config.num_segments//10
                _, topk_indices_action = torch.topk(action, K1, dim=1)
                background = 1 - action    # [16, 750, 1]
                _, topk_indices_background = torch.topk(background, K2, dim=1)

                cls_agnostic_gt = self.calculate_pesudo_target1(batch_size, topk_indices_action)
                cls_agnostic_gt_rev = self.calculate_pesudo_target1(batch_size, topk_indices_background)

                cas_top_action = torch.sum(action * cls_agnostic_gt.permute(0,2,1),dim=1) / K1        
                cas_top_background = torch.sum(background * cls_agnostic_gt_rev.permute(0,2,1),dim=1) / K2   
                cas_top_ca = torch.cat((cas_top_action, cas_top_background),dim=-1)    

                # losses
                cost = self.calculate_all_losses(epoch, cas_top, cas_top_fg, cas_top_fp, _label, action_flow, action_rgb, discrim, cas_top_ca)

                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                if epoch > 35 :               # 35
                    self.evaluate(epoch=epoch)


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
