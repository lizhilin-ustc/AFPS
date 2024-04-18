import os
import sys
import math
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.interpolate import interp1d
from tqdm import tqdm


def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, lambda_=0.25, gamma=0.2):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - lambda_ * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + lambda_ * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])

        temp.append(c_temp)
    return temp


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def basnet_nms(proposals, thresh, soft_nms=False, nms_alpha=0):
    proposals = np.array(proposals)

    x1 = proposals[:, 2]        # start
    x2 = proposals[:, 3]        # end
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]    #　[::-1] 顺序倒置

    keep = []
    not_keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        if soft_nms:
            inv_inds = np.where(iou >= thresh)[0]
            props_mod = proposals[order[inv_inds + 1]]

            for k in range(props_mod.shape[0]):
                props_mod[k, 1] = props_mod[k, 1] * np.exp(-np.square(iou[inv_inds][k]) / nms_alpha)

            not_keep.extend(props_mod.tolist())

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    if soft_nms:
        keep.extend(not_keep)
    # print(np.array(keep).shape)
    # exit()

    return keep


def instance_selection_function(cas, *actionness):
    return (cas + sum(actionness)) / (1 + len(actionness))

def instance_selection_function1(a, b, c):
    return (a + b + c) / 2

def instance_selection_function2(cas_r, cas_f, cas_flow, cas_rgb):
    combine_cas = (0.5*cas_r +0.5*cas_f + cas_flow + cas_rgb)/3
    return combine_cas


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def result2json(result):
    from config.config_thumos import class_dict
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': class_dict[result[i][j][0]], 'score': result[i][j][1],   #####
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file

###############
def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])
    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta
    ret[ret > 1] = 1
    ret[ret < 0] = 0
    return ret

def get_pred_activations(src, pred, UP_SCALE):
    src = minmax_norm(src)
    if len(src.size()) == 2:
        src = src.repeat((750, 1, 1)).permute(1, 2, 0)
    src_pred = src[0].cpu().numpy()[:, pred]
    src_pred = np.reshape(src_pred, (src.size(1), -1, 1))
    src_pred = upgrade_resolution(src_pred, UP_SCALE)
    return src_pred

def get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, config):
    prop_dict = {}
    for th in config.act_thresh:
        cas_tmp = cas_pred.copy()
        num_segments = cas_pred.shape[0]//config.scale
        cas_tmp[cas_tmp[:, :, 0] < th] = 0
        seg_list = [np.where(cas_tmp[:, c, 0] > 0) for c in range(len(pred))]
        proposals = get_proposal_oic(seg_list, cas_tmp, score_np, pred, config.scale, vid_num_seg, config.feature_fps, num_segments)
        for i in range(len(proposals)):
            class_id = proposals[i][0][0]
            prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]

    for th in config.act_thresh1:
        aness_tmp = aness_pred.copy()
        num_segments = aness_pred.shape[0]//config.scale
        aness_tmp[aness_tmp[:, :, 0] < th] = 0
        seg_list = [np.where(aness_tmp[:, c, 0] > 0) for c in range(len(pred))]
        proposals = get_proposal_oic(seg_list, cas_pred, score_np, pred, config.scale, \
                        vid_num_seg, config.feature_fps, num_segments)
        for i in range(len(proposals)):
            class_id = proposals[i][0][0]
            prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]
    return prop_dict

def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep
