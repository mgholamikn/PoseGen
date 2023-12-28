## how to use the code:
# nohup python render_3dpw_testset.py --nerf_args configs/surreal/surreal.txt --ckptpath logs/surreal_model/surreal.tar  --dataset surreal --entry hard  --runname render_3dpw_testset --white_bkgd  --render_res 512 512 > render_3dpw_testset.out 2>&1 &
from __future__ import absolute_import
from email.mime import image
import os
import torch
import shutil
import imageio
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import torch.distributions as dist
import numpy as np
import cv2
from os.path import join


from HybrIK.hybrik.utils.config import update_config
import deepdish as dd
from run_nerf import render_path
from run_nerf import config_parser as nerf_config_parser
from core.imutils import *
from core.pose_opt import load_poseopt_from_state_dict, pose_ckpt_to_pose_data
from core.load_data import generate_bullet_time, get_dataset
from core.raycasters import create_raycaster
from os.path import join
from core.utils.evaluation_helpers import txt_to_argstring
from core.utils.skeleton_utils import CMUSkeleton, get_smpl_l2ws_torch, nerf_c2w_to_extrinsic_torch, nerf_extrinsic_to_c2w_torch, smpl_rest_pose, get_smpl_l2ws, get_per_joint_coords
from core.utils.skeleton_utils import draw_skeletons_3d, rotate_x, rotate_y, axisang_to_rot, rot_to_axisang
from HybrIK.hybrik.utils.presets import SimpleTransform3DSMPLCam
from pytorch_msssim import SSIM
import torch.nn as nn
import time
from progress.bar import Bar
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
# import torchgeometry as tgm
import pytorch3d.transforms as torch3d
import os.path as path
import datetime
from tensorboardX import SummaryWriter
import copy
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader,Dataset
from skimage.transform import rescale, resize, downscale_local_mean
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms as T
import gc
gc.collect()
torch.cuda.empty_cache()
det_transform = T.Compose([T.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import h5py
GT_PREFIXES = {
    'h36m': 'data/h36m/',
    'surreal': None,
    'perfcap': 'data/',
    'mixamo': 'data/mixamo',
}

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    # Training detail
    parser.add_argument('--epochs', default=4, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--decay_epoch', default=0, type=int, metavar='N', help='number of decay epochs')

    # Learning rate
    parser.add_argument('--lr_g', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for augmentor/generator')
    parser.add_argument('--lr_d', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for discriminator')
    parser.add_argument('--lr_p', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for posenet')
    parser.add_argument("--lr_spin", type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.add_argument('--df', default=2, type=int, help='update discriminator frequency')
    parser.set_defaults(max_norm=True)
    # nerf config
    parser.add_argument('--nerf_args', type=str, required=True,
                        help='path to nerf configuration (args.txt in log)')
    parser.add_argument('--ckptpath', type=str, required=True,
                        help='path to ckpt')

    # render config
    parser.add_argument('--render_res', nargs='+', type=int, default=[1000, 1000],
                        help='tuple of resolution in (H, W) for rendering')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset to render')
    parser.add_argument('--entry', type=str, required=True,
                        help='entry in the dataset catalog to render')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='render with white background')
    parser.add_argument('--render_type', type=str, default='retarget',
                        help='type of rendering to conduct')
    parser.add_argument('--save_gt', action='store_true',
                        help='save gt frames')
    parser.add_argument('--fps', type=int, default=14,
                        help='fps for video')
    parser.add_argument('--rpi', type=int, default=20,
                        help='number of images to render per iteration')
    parser.add_argument('--mesh_res', type=int, default=255,
                        help='resolution for marching cubes')
    # kp-related
    parser.add_argument('--render_refined', action='store_true',
                        help='render from refined poses')
    parser.add_argument('--subject_idx', type=int, default=0,
                        help='which subject to render (for MINeRF)')

    # frame-related
    parser.add_argument('--selected_idxs', nargs='+', type=int, default=None,
                        help='hand-picked idxs for rendering')
    parser.add_argument('--selected_framecode', type=int, default=None,
                        help='hand-picked framecode for rendering')

    # saving
    parser.add_argument('--outputdir', type=str, default='/media/ExtHDD/Mohsen_Data/NerfPose/render_output/',
                        help='output directory')
    parser.add_argument('--runname', type=str, required=True,
                        help='run name as an identifier ')

    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='to do evaluation at the end or not (only in bounding box)')

    parser.add_argument('--no_save', action='store_true',
                        help='no image saving operation')

    return parser

def load_nerf(args, nerf_args, skel_type=CMUSkeleton):

    ckptpath = args.ckptpath
    nerf_args.ft_path = args.ckptpath

    # some info are unknown/not provided in nerf_args
    # dig those out from state_dict directly
    nerf_sdict = torch.load(ckptpath)

    # get data_attrs used for training the models
    data_attrs = get_dataset(nerf_args).get_meta()
    if 'framecodes.codes.weight' in nerf_sdict['network_fn_state_dict']:
        framecodes = nerf_sdict['network_fn_state_dict']['framecodes.codes.weight']
        data_attrs['n_views'] = framecodes.shape[0]

    # load poseopt_layer (if exist)
    popt_layer = None
    if nerf_args.opt_pose:
        popt_layer = load_poseopt_from_state_dict(nerf_sdict)
    nerf_args.finetune = True

    render_kwargs_train, render_kwargs_test, _, grad_vars, _, _ = create_raycaster(nerf_args, data_attrs, device=device)

    # freeze weights
    for grad_var in grad_vars:
        grad_var.requires_grad = False

    render_kwargs_test['ray_caster'] = render_kwargs_train['ray_caster']
    render_kwargs_test['ray_caster'].eval()

    return render_kwargs_test, popt_layer

def load_render_data(args, bones,c2ws,nerf_args, poseopt_layer=None, opt_framecode=True):
    # TODO: note that for models trained on SPIN data, they may not react well
    catalog = init_catalog(args)[args.dataset][args.entry]
    render_data = catalog.get(args.render_type, {})
    data_h5 = catalog['data_h5']

    # to real cameras (due to the extreme focal length they were trained on..)
    # TODO: add loading with opt pose option
    if poseopt_layer is not None:
        rest_pose = poseopt_layer.get_rest_pose().cpu().numpy()[0]
        # print("Load rest pose for poseopt!")
    else:
        try:
            rest_pose = dd.io.load(data_h5, '/rest_pose')
            # print("Load rest pose from h5!")
        except:
            rest_pose = smpl_rest_pose * dd.io.load(data_h5, '/ext_scale')
            # print("Load smpl rest pose!")


    if args.render_refined:
        if 'refined' in catalog:
            # print(f"loading refined poses from {catalog['refined']}")
            #poseopt_layer = load_poseopt_from_state_dict(torch.load(catalog['refined']))
            kps, bones = pose_ckpt_to_pose_data(catalog['refined'], legacy=True)[:2]
        else:
            with torch.no_grad():
                bones = poseopt_layer.get_bones().cpu().numpy()
                kps = poseopt_layer(np.arange(len(bones)))[0].cpu().numpy()

        if render_data is not None:
            render_data['refined'] = [kps, bones]
            render_data['idx_map'] = catalog.get('idx_map', None)
        else:
            render_data = {'refined': [kps, bones],
                           'idx_map': catalog['idx_map']}  
    else:
        render_data['idx_map'] = catalog.get('idx_map', None)

    pose_keys = ['/kp3d', '/bones']
    cam_keys = ['/c2ws', '/focals']
    # Do partial load here!
    # Need:
    # 1. kps: for root location
    # 2. bones: for bones
    # 3. camera stuff: focals, c2ws, ... etc
    _, focals = dd.io.load(data_h5, cam_keys)
    _, H, W, _ = dd.io.load(data_h5, ['/img_shape'])[0]
    
    # handel resolution
    if args.render_res is not None:
        assert len(args.render_res) == 2, "Image resolution should be in (H, W)"
        H_r, W_r = args.render_res
        # TODO: only check one side for now ...
        scale = float(H_r) / float(H)
        focals *= scale
        H, W = H_r, W_r
    
    # Load data based on type:
    # bones = None
    bg_imgs, bg_indices = None, None
    if args.render_type in ['retarget', 'mesh']:
        # print(f'Load data for retargeting!')
        kps, skts, c2ws, cam_idxs, focals, bones = load_retarget(data_h5, bones,c2ws, focals,
                                                                 rest_pose, pose_keys,
                                                                 **render_data)
    

    gt_paths, gt_mask_paths = None, None
    is_gt_paths = True
    

    subject_idxs = None
    if nerf_args.nerf_type.startswith('minerf'):
        subject_idxs = (np.ones((len(kps),)) * args.subject_idx).astype(np.int64)

    ret_dict = {'kp': kps, 'skts': skts, 'render_poses': c2ws,
                'cams': cam_idxs if opt_framecode else None,
                'hwf': (H, W, focals),
                'bones': bones,
                'bg_imgs': bg_imgs,
                'bg_indices': bg_indices,
                'subject_idxs': subject_idxs}


    return ret_dict

def init_catalog(args, n_bullet=10):

    RenderCatalog = {
        'h36m': None,
        'surreal': None,
        'perfcap': None,
        'mixamo': None,
        '3dhp': None,
    }

    def load_idxs(path):
        if not os.path.exists(path):
            # print(f'Index file {path} does not exist.')
            return []
        return np.load(path)
    def set_dict(selected_idxs, **kwargs):
        return {'selected_idxs': np.array(selected_idxs), **kwargs}

    # H36M
    s9_idx = np.arange(20)*20
    h36m_s9 = {
        'data_h5': 'data/h36m/S9_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/h36m/s9_sub64_500k.tar',
        'retarget': set_dict(s9_idx, length=5),
        'bullet': set_dict(s9_idx, n_bullet=n_bullet, undo_rot=True,
                           center_cam=True),
        'interpolate': set_dict(s9_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'correction': set_dict(load_idxs('data/h36m/S9_top50_refined.npy')[:1], n_step=30),
        'animate': set_dict([1000, 1059, 2400], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([17,19,21,23])),
        'bubble': set_dict(s9_idx, n_step=30),
        'poserot': set_dict(np.array([1000])),
        'val': set_dict(load_idxs('data/h36m/S9_val_idxs.npy'), length=1, skip=1),
    }
    s11_idx = [213, 656, 904, 1559, 1815, 2200, 2611, 2700, 3110, 3440, 3605]
    h36m_s11 = {
        'data_h5': 'data/h36m/S11_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/h36m/s11_sub64_500k.tar',
        'retarget': set_dict(s11_idx, length=5),
        'bullet': set_dict(s11_idx, n_bullet=n_bullet, undo_rot=True,
                           center_cam=True),
        'interpolate': set_dict(s11_idx, n_step=10, undo_rot=True,
                                center_cam=True),

        'correction': set_dict(load_idxs('data/h36m/S11_top50_refined.npy')[:1], n_step=30),
        'animate': set_dict([2507, 700, 900], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([3,6,9,12,15,16,18])),
        'bubble': set_dict(s11_idx, n_step=30),
        'val': set_dict(load_idxs('data/h36m/S11_val_idxs.npy'), length=1, skip=1),
    }

    # SURREAL
    easy_idx = [10, 70, 350, 420, 490, 910, 980, 1050]
    surreal_val = {
        'data_h5': 'data/surreal/surreal_val_h5py.h5',
        'val': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy'), length=1, skip=1),
        'val2': set_dict(load_idxs('data/surreal/surreal_val_idxs.npy')[:300], length=1, skip=1),
    }
    surreal_easy = {
        'data_h5': 'data/surreal/surreal_train_h5py.h5',
        'retarget': set_dict(easy_idx, length=25, skip=2, center_kps=True),
        'bullet': set_dict(easy_idx, n_bullet=n_bullet),
        'bubble': set_dict(easy_idx, n_step=30),
    }
    hard_idx = [140, 210, 280, 490, 560, 630, 700, 770, 840, 910]
    surreal_hard = {
        'data_h5': 'data/surreal/surreal_train_h5py.h5',
        'retarget': set_dict(hard_idx, length=60, skip=5, center_kps=True),
        'bullet': set_dict([190,  210,  230,  490,  510,  530,  790,  810,  830,  910,  930, 950, 1090, 1110, 1130],
                           n_bullet=n_bullet, center_kps=True, center_cam=False),
        'bubble': set_dict(hard_idx, n_step=30),
        'val': set_dict(np.array([1200 * i + np.arange(420, 700)[::5] for i in range(0, 9, 2)]).reshape(-1), length=1, skip=1),
        'mesh': set_dict([930], length=1, skip=1),
    }

    # PerfCap
    weipeng_idx = [0, 50, 100, 150, 200, 250, 300, 350, 430, 480, 560,
                   600, 630, 660, 690, 720, 760, 810, 850, 900, 950, 1030,
                   1080, 1120]
    perfcap_weipeng = {
        'data_h5': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/perfcap/weipeng_tv_500k.tar',
        'retarget': set_dict(weipeng_idx, length=30, skip=2),
        'bullet': set_dict(weipeng_idx, n_bullet=n_bullet),
        'interpolate': set_dict(weipeng_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(weipeng_idx, n_step=30),
        'val': set_dict(np.arange(1151)[-230:], length=1, skip=1),
        'animate': set_dict([300, 480, 700], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([1,4,7,10,17,19,21,23])),
    }

    nadia_idx = [0, 65, 100, 125, 230, 280, 410, 560, 600, 630, 730, 770,
                 830, 910, 1010, 1040, 1070, 1100, 1285, 1370, 1450, 1495,
                 1560, 1595]
    perfcap_nadia = {
        'data_h5': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor_processed_h5py.h5',
        'refined': 'neurips21_ckpt/trained/ours/perfcap/nadia_tv_500k.tar',
        'retarget': set_dict(nadia_idx, length=30, skip=2),
        'bullet': set_dict(nadia_idx, n_bullet=n_bullet),
        'interpolate': set_dict(nadia_idx, n_step=10, undo_rot=True,
                                center_cam=True, center_kps=True),
        'bubble': set_dict(nadia_idx, n_step=30),
        'animate': set_dict([280, 410, 1040], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([1,2,4,5,7,8,10,11])),
        'val': set_dict(np.arange(1635)[-327:], length=1, skip=1),
    }

    # Mixamo
    james_idx = [20, 78, 138, 118, 1149, 333, 3401, 2221, 4544]
    mixamo_james = {
        'data_h5': 'data/mixamo/James_processed_h5py.h5',
        'idx_map': load_idxs('data/mixamo/James_selected.npy'),
        'refined': 'neurips21_ckpt/trained/ours/mixamo/james_tv_500k.tar',
        'retarget': set_dict(james_idx, length=30, skip=2),
        'bullet': set_dict(james_idx, n_bullet=n_bullet, center_cam=True, center_kps=True),
        'interpolate': set_dict(james_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(james_idx, n_step=30),
        'animate': set_dict([3401, 1149, 4544], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([18,19,20,21,22,23])),
        'mesh': set_dict([20, 78], length=1, undo_rot=False),
    }

    archer_idx = [158, 672, 374, 414, 1886, 2586, 2797, 4147, 4465]
    mixamo_archer = {
        'data_h5': 'data/mixamo/Archer_processed_h5py.h5',
        'idx_map': load_idxs('data/mixamo/Archer_selected.npy'),
        'refined': 'neurips21_ckpt/trained/ours/mixamo/archer_tv_500k.tar',
        'retarget': set_dict(archer_idx, length=30, skip=2),
        'bullet': set_dict(archer_idx, n_bullet=n_bullet, center_cam=True, center_kps=True),
        'interpolate': set_dict(archer_idx, n_step=10, undo_rot=True,
                                center_cam=True),
        'bubble': set_dict(archer_idx, n_step=30),
        'animate': set_dict([1886, 2586, 4465], n_step=10, center_cam=True, center_kps=True,
                            joints=np.array([18,19,20,21,22,23])),
    }

    # NeuralBody
    nb_subjects = ['315', '377', '386', '387', '390', '392', '393', '394']
    # TODO: hard-coded: 6 views
    nb_idxs = np.arange(len(np.concatenate([np.arange(1, 31), np.arange(400, 601)])) * 6)
    nb_dict = lambda subject: {'data_h5': f'data/zju_mocap/{subject}_test_h5py.h5',
                               'val': set_dict(nb_idxs, length=1, skip=1)}

    RenderCatalog['h36m'] = {
        'S9': h36m_s9,
        'S11': h36m_s11,
        'gt_to_mask_map': ('imageSequence', 'Mask'),
    }
    RenderCatalog['surreal'] = {
        'val': surreal_val,
        'easy': surreal_easy,
        'hard': surreal_hard,
    }
    RenderCatalog['perfcap'] = {
        'weipeng': perfcap_weipeng,
        'nadia': perfcap_nadia,
        'gt_to_mask_map': ('images', 'masks'),
    }
    RenderCatalog['mixamo'] = {
        'james': mixamo_james,
        'archer': mixamo_archer,
    }
    RenderCatalog['neuralbody'] = {
        f'{subject}': nb_dict(subject) for subject in nb_subjects
    }

    return RenderCatalog

def find_idxs_with_map(selected_idxs, idx_map):
    if idx_map is None:
        return selected_idxs
    match_idxs = []
    for sel in selected_idxs:
        for i, m in enumerate(idx_map):
            if m == sel:
                match_idxs.append(i)
                break
    return np.array(match_idxs)


def load_retarget(pose_h5,bones, c2ws, focals, rest_pose, pose_keys,
                  selected_idxs, length, skip=1, refined=None,
                  center_kps=False, idx_map=None, is_surreal=False,
                  undo_rot=False, is_neuralbody=False):

    # l = length
    # if skip > 1 and l > 1:
    #     selected_idxs = np.concatenate([np.arange(s, min(s+l, len(c2ws)))[::skip] for s in selected_idxs])
    # #selected_idxs = np.clip(selected_idxs, a_min=0, a_max=len(c2ws)-1)

    # # c2ws = c2ws[selected_idxs]
    # # if isinstance(focals, float):
    # #     focals = np.array([focals] * len(selected_idxs))
    # # else:
    # #     focals = focals[selected_idxs]
    # cam_idxs = selected_idxs

    # if refined is None:
    #     if not is_surreal and not is_neuralbody:
    #         kps, bones = dd.io.load(pose_h5, pose_keys, sel=dd.aslice[selected_idxs, ...])
    #     elif is_neuralbody:
    #         kps, bones = dd.io.load(pose_h5, pose_keys)
    #         # TODO: hard-coded, could be problematic
    #         kps = kps.reshape(-1, 1, 24, 3).repeat(6, 1).reshape(-1, 24, 3)
    #         bones = bones.reshape(-1, 1, 24, 3).repeat(6, 1).reshape(-1, 24, 3)
    #     else:
    #         kps, bones = dd.io.load(pose_h5, pose_keys)
    #         kps = kps[None].repeat(9, 0).reshape(-1, 24, 3)[selected_idxs]
    #         bones = bones[None].repeat(9, 0).reshape(-1, 24, 3)[selected_idxs]
    #     selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    # else:
    #     selected_idxs = find_idxs_with_map(selected_idxs, idx_map)
    #     kps, bones = refined
    #     kps = kps[selected_idxs]
    #     bones = bones[selected_idxs]

    # if center_kps:
    #     root = kps[..., :1, :].copy() # assume to be CMUSkeleton
    #     kps[..., :, :] -= root

    # if undo_rot:
    #     bones[..., 0, :] = np.array([1.5708, 0., 0.], dtype=np.float32).reshape(1, 1, 3)
    
    bones=bones
    cam_idxs=np.arange(bones.shape[0])
    focals = focals[cam_idxs]
    c2ws = c2ws[cam_idxs]
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
    # l2ws[..., :3, -1] += kps[..., :1, :].copy()
    kps = l2ws[..., :3, -1]
    skts = np.linalg.inv(l2ws)

    return kps, skts, c2ws, cam_idxs, focals, bones


def to_tensors(data_dict):
    tensor_dict = {}
    for k in data_dict:
        if isinstance(data_dict[k], np.ndarray):
            if k == 'bg_indices' or k == 'subject_idxs':
                tensor_dict[k] = torch.tensor(data_dict[k]).long()
            else:
                tensor_dict[k] = torch.tensor(data_dict[k]).float()
        elif k == 'hwf' or k == 'cams' or k == 'bones':
            tensor_dict[k] = data_dict[k]
        elif data_dict[k] is None:
            tensor_dict[k] = None
        else:
            raise NotImplementedError(f"{k}: only nparray and hwf are handled now!")
    return tensor_dict


# self define tools
class Summary(object):
    def __init__(self, directory):
        self.directory = directory
        self.epoch = 0
        self.writer = None
        self.phase = 0
        self.train_iter_num = 0
        self.train_realpose_iter_num = 0
        self.train_fakepose_iter_num = 0
        self.test_iter_num = 0
        self.test_MPI3D_iter_num = 0

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def summary_train_iter_num_update(self):
        self.train_iter_num = self.train_iter_num + 1

    def summary_train_realpose_iter_num_update(self):
        self.train_realpose_iter_num = self.train_realpose_iter_num + 1

    def summary_train_fakepose_iter_num_update(self):
        self.train_fakepose_iter_num = self.train_fakepose_iter_num + 1

    def summary_test_iter_num_update(self):
        self.test_iter_num = self.test_iter_num + 1

    def summary_test_MPI3D_iter_num_update(self):
        self.test_MPI3D_iter_num = self.test_MPI3D_iter_num + 1

    def summary_epoch_update(self):
        self.epoch = self.epoch + 1

    def summary_phase_update(self):
        self.phase = self.phase + 1

class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath,  args, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

        self.record_args(args)

    def record_args(self, args):
        self.file.write(str(args))
        self.file.write('\n')
        self.file.flush()

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

# To store data in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=4096):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

from torch.optim import lr_scheduler
def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

def swap_mat(mat):
    # [right, -up, -forward]
    # equivalent to right multiply by:
    # [1, 0, 0, 0]
    # [0,-1, 0, 0]
    # [0, 0,-1, 0]
    # [0, 0, 0, 1]
    return np.concatenate([
        mat[..., 0:1], -mat[..., 1:2], -mat[..., 2:3], mat[..., 3:]
        ], axis=-1)

def nerf_c2w_to_extrinsic(c2w):
    return np.linalg.inv(swap_mat(c2w))


def coord_to_homogeneous(c):
    assert c.shape[-1] == 3

    if len(c.shape) == 2:
        h = np.ones((c.shape[0], 1)).astype(c.dtype)
        return np.concatenate([c, h], axis=1)
    elif len(c.shape) == 1:
        h = np.array([0, 0, 0, 1]).astype(c.dtype)
        h[:3] = c
        return h
    else:
        raise NotImplementedError(f"Input must be a 2-d or 1-d array, got {len(c.shape)}")


def focal_to_intrinsic_np(focal):
    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal
    return np.array([[focal_x,      0, 0, 0],
                     [     0, focal_y, 0, 0],
                     [     0,       0, 1, 0]],
                    dtype=np.float32)

def world_to_cam(pts, extrinsic, H, W, focal, center=None):

    if center is None:
        offset_x = W * .5
        offset_y = H * .5
    else:
        offset_x, offset_y = center

    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)

    intrinsic = focal_to_intrinsic_np(focal)

    cam_pts = pts @ extrinsic.T @ intrinsic.T
    cam_pts = cam_pts[..., :2] / cam_pts[..., 2:3]
    cam_pts[cam_pts == np.inf] = 0.
    cam_pts[..., 0] += offset_x
    cam_pts[..., 1] += offset_y
    return cam_pts

def skeleton3d_to_2d(kps, c2ws, H, W, focals, centers=None):

    exts = np.array([nerf_c2w_to_extrinsic(c2w) for c2w in c2ws])

    kp2ds = []
    for i, (kp, ext) in enumerate(zip(kps, exts)):
        f = focals[i] if not isinstance(focals, float) else focals
        h = H if isinstance(H, int) else H[i]
        w = W if isinstance(W, int) else W[i]
        center = centers[i] if centers is not None else None
        kp2d = world_to_cam(kp, ext, h, w, f, center)
        kp2ds.append(kp2d)

    return np.array(kp2ds)


def project_to_2d(kps, exts, H, W, focals, centers=None):
    kps,exts,H,W,focals=torch.as_tensor(kps),torch.as_tensor(exts),torch.as_tensor(H),torch.as_tensor(W),torch.as_tensor(focals)
    
    
    offset_x = W * .5
    offset_y = H * .5

    # exts=np.repeat(np.expand_dims(exts,axis=0),kps.shape[0],axis=0)
    kps_homogeneous=torch.ones(kps.shape[0],kps.shape[1],4).to(device)
    kps_homogeneous[:,:,:3]=kps
    intrinsics=torch.zeros(kps.shape[0],3,4).to(device)
    intrinsics[:,0,0]=focals[0]
    intrinsics[:,1,1]=focals[1]
    intrinsics[:,2,2]=1

    

    intrinsics=intrinsics.float()
    kps_homogeneous=kps_homogeneous.float()
    exts=exts.float()
    kp3ds=kps_homogeneous@exts.transpose(1,2)
    kp2ds=kp3ds@intrinsics.transpose(1,2)
    kp2ds=kp2ds[...,:2]/kp2ds[...,2:3]
    kp2ds[kp2ds == np.inf] = 0.
    kp2ds[..., 0] += offset_x
    kp2ds[..., 1] += offset_y

    return kp2ds, kp3ds[:,:,:3]

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - torch.tensor([1, h / w]).to(device)

class Linear(nn.Module):
    def __init__(self, linear_size):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(inplace=True)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)

        return y


######################################################
###################  START  ##########################
######################################################
class PoseGenerator(nn.Module):
    def __init__(self, args, input_size=24 * 3):
        super(PoseGenerator, self).__init__()
        self.BAprocess = BAGenerator(input_size=24 * 3,noise_channle=32)
        # self.BLprocess = BLGenerator(input_size=16 * 3, blr_tanhlimit=args.blr_tanhlimit)
        self.RTprocess = RTGenerator(input_size=24 * 3) #target

    def forward(self, inputs_3d):
        '''
        input: 3D pose
        :param inputs_3d: nx16x3, with hip root
        :return: nx16x3
        '''
        
        pose_ba = self.BAprocess(inputs_3d)  # diff may be used for div loss
        # pose_bl, blr = self.BLprocess(inputs_3d, pose_ba)  # blr used for debug
        R,T,pose_rt = self.RTprocess(inputs_3d)  # rt=(r,t) used for debug

        return {'pose_ba': pose_ba,
                'ba_diff': None,
                'pose_bl': None,
                'blr': None,
                'pose_rt': pose_rt,
                'R': R,
                'T': T}


######################################################
###################  END  ############################
######################################################

class BAGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=72, linear_size=256, num_stage=2, p_dropout=0.5):
        super(BAGenerator, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 24 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.noise_channle, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size,24*4) #*2+(self.input_size-3)//3

        self.relu = nn.LeakyReLU(inplace=True)

    def Sampler(self,dim0,device):
        # Define the parameters for the Gaussian modes
        num_modes = 5
        means = [torch.ones(self.noise_channle)*-10, torch.ones(self.noise_channle)*-5, torch.zeros(self.noise_channle), torch.ones(self.noise_channle)*5, torch.ones(self.noise_channle)*10]
        covariances = torch.eye(self.noise_channle) #torch.tensor([[[0.5, 0.2], [0.2, 0.5]]]*num_modes)

        # Create a list of Gaussian distributions with different means and covariances
        gaussians = [dist.MultivariateNormal(means[i], covariance_matrix=covariances,device=device) for i in range(num_modes)]

        # Sample from each Gaussian distribution
        samples_per_mode = dim0
        samples = torch.cat([gaussian.sample([samples_per_mode]) for gaussian in gaussians], dim=0)
        idxs=torch.randperm(samples.shape[0])
        samples=samples[idxs]
        return samples[:dim0]


    def forward(self, inputs_3d):
        '''
        :param inputs_3d: nx16x3.
        :return: nx16x3
        '''
            
        # pre-processing

        noise1 = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device) #*2-1
        noise2 = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device)+5
        noise3 = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device)+10
        noise4 = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device)-5
        noise5 = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device)-10
        noise  = torch.cat([noise1,noise2,noise3,noise4,noise5],dim=0)
        noise  = noise[torch.randperm(len(noise))]
        noise  = noise[:inputs_3d.shape[0]]
        # noise=self.Sampler(inputs_3d.shape[0],device=inputs_3d.device)
        # noise = noise / noise.norm(dim=1, keepdim=True)

        y = self.w1(noise) #torch.cat((bones_vec, noise), dim=-1)
  
        y = self.batch_norm1(y)

        y = self.relu(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = y.view(inputs_3d.size(0), -1, 4)

        y_axis=y[:,:,:3]

        y_axis = y_axis/torch.linalg.norm(y_axis,dim=-1,keepdim=True)
        y_theta =y[:,:,3:4]
        y_theta=y_theta

        out = y_axis*y_theta
        out[:,0]*=3.14*2
        
        # y_theta=y[:,:,6:7]
        # y_mean=y[:,:,:3]
        # y_std=y[:,:,3:6]*y[:,:,3:6]
        # y_axis = torch.normal(mean=y_mean,std=y_std)
        # y_axis = y_axis/torch.linalg.norm(y_axis,dim=-1,keepdim=True)
        # out = y_axis*y_theta
        # out[:,0]*=3.14*2

        return out


class RTGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=72, linear_size=256, num_stage=2, p_dropout=0.5):
        super(RTGenerator, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size -> for R
        self.w1_R = nn.Linear(self.noise_channle, self.linear_size)
        self.batch_norm_R = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_R = []
        for l in range(num_stage):
            self.linear_stages_R.append(Linear(self.linear_size))
        self.linear_stages_R = nn.ModuleList(self.linear_stages_R)

        # process input to linear size -> for T
        self.w1_T = nn.Linear(self.noise_channle, self.linear_size) 
        self.batch_norm_T = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_T = []
        for l in range(num_stage):
            self.linear_stages_T.append(Linear(self.linear_size))
        self.linear_stages_T = nn.ModuleList(self.linear_stages_T)

        # post processing

        self.w2_R = nn.Linear(self.linear_size, 7)
        self.w2_T = nn.Linear(self.linear_size, 3) 

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,inputs_3d):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # caculate R
        noise = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device)
        r = self.w1_R(noise) #torch.cat((x, noise), dim=1)
        r = self.batch_norm_R(r)
        r = self.relu(r)
        for i in range(self.num_stage):
            r = self.linear_stages_R[i](r)

        # r = self.w2_R(r)
        r_mean=r[:,:3]
        r_std=r[:,3:6]*r[:,3:6]
        r_axis = torch.normal(mean=r_mean,std=r_std)
        r_axis = r_axis/torch.linalg.norm(r_axis,dim=-1,keepdim=True)
        r_axis = r_axis*r[:,6:7]

        rM=torch3d.axis_angle_to_matrix(r_axis) #axis_angle
        # rM = torch3d.euler_angles_to_matrix(r_axis,["Z","Y","X"])  #euler_angle
        # rM= torch3d.quaternion_to_matrix(r_axis) #quaternion
        

        # caculate T
        noise = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device)
        t = self.w1_T(noise) #torch.cat((x, noise), dim=1)
        t = self.batch_norm_T(t)
        t = self.relu(t)
        for i in range(self.num_stage):
            t = self.linear_stages_T[i](t)

        t = self.w2_T(t)

        t[:, 2] = t[:, 2].clone() * t[:, 2].clone()
        t = t.view(inputs_3d.shape[0],1, 3)  # Nx1x3 translation t
        inputs_3d=inputs_3d-inputs_3d[:,:1,:]
        inputs_3d=inputs_3d.permute(0,2,1).contiguous()
     
        out=torch.matmul(rM,inputs_3d)
        out=out.permute(0,2,1).contiguous()
        out=out+t
        return rM, t[:,0] , out # return r t for debug

class Disc_Joint_Path(nn.Module):
    def __init__(self, num_joints=1, channel=500, channel_mid=1000):
        super(Disc_Joint_Path, self).__init__()
        # KCS path
        self.layer_1 = nn.Linear(num_joints*3, channel)
        self.layer_2 = nn.Linear(channel, channel)
        self.layer_3 = nn.Linear(channel, channel)
        self.layer_last = nn.Linear(channel, channel_mid)
        self.layer_pred = nn.Linear(channel_mid, 1)
        self.relu = nn.LeakyReLU()

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # KCS path
        x=self.relu(self.layer_1(x))
        x=self.relu(self.layer_2(x))
        x=self.relu(self.layer_3(x))
        x=self.relu(self.layer_last(x))
        y=self.layer_pred(x)
        return y
class Pos3dDiscriminator(nn.Module):
    def __init__(self,num_joints=24,channel=1000,channel_mid=100):
        super(Pos3dDiscriminator,self).__init__()

        self.layer_left_leg = Disc_Joint_Path(num_joints=3)
        self.layer_right_leg = Disc_Joint_Path(num_joints=3)
        self.layer_left_arm = Disc_Joint_Path(num_joints=6)
        self.layer_right_arm  = Disc_Joint_Path(num_joints=6)
        self.layer_head  = Disc_Joint_Path(num_joints=3)
        self.layer_torso  = Disc_Joint_Path(num_joints=10)
        self.layer_full_body  = Disc_Joint_Path(num_joints=24)
        # self.layer_7 = Disc_Joint_Path()
        # self.layer_8 = Disc_Joint_Path()
        # self.layer_9 = Disc_Joint_Path()
        # self.layer_10  = Disc_Joint_Path()
        # self.layer_11 = Disc_Joint_Path()
        # self.layer_12 = Disc_Joint_Path()
        # self.layer_13 = Disc_Joint_Path()
        # self.layer_14 = Disc_Joint_Path()
        # self.layer_15 = Disc_Joint_Path()
        # self.layer_16  = Disc_Joint_Path()
        # self.layer_17  = Disc_Joint_Path()
        # self.layer_18  = Disc_Joint_Path()
        # self.layer_19 = Disc_Joint_Path()
        # self.layer_20  = Disc_Joint_Path()
        # self.layer_21  = Disc_Joint_Path()
        # self.layer_22  = Disc_Joint_Path()
        # self.layer_23  = Disc_Joint_Path()
        # self.layer_24  = Disc_Joint_Path()

    def forward(self, input_3d):
        # input_3d=input_3d.reshape(-1,24*3)
        x1= self.layer_left_leg(input_3d[:,[4,7,10]].reshape(-1,3*3))
        x2= self.layer_right_leg(input_3d[:,[5,8,11]].reshape(-1,3*3))
        x3= self.layer_left_arm(input_3d[:,[9,13,16,18,20,22]].reshape(-1,6*3))
        x4= self.layer_right_arm(input_3d[:,[9,14,17,19,21,23]].reshape(-1,6*3))
        x5= self.layer_torso(input_3d[:,[0,1,2,3,6,9,13,14,16,17]].reshape(-1,10*3))
        x6= self.layer_head(input_3d[:,[9,12,15]].reshape(-1,3*3))
        x7= self.layer_full_body(input_3d.reshape(-1,24*3))
        # x7= self.layer_7(input_3d[:,6])
        # x8= self.layer_8(input_3d[:,7])
        # x9= self.layer_9(input_3d[:,8])
        # x10= self.layer_10(input_3d[:,9])
        # x11= self.layer_11(input_3d[:,10])
        # x12= self.layer_12(input_3d[:,11])
        # x13= self.layer_13(input_3d[:,12])
        # x14= self.layer_14(input_3d[:,13])
        # x15= self.layer_15(input_3d[:,14])
        # x16= self.layer_16(input_3d[:,15])
        # x17= self.layer_17(input_3d[:,16])
        # x18= self.layer_18(input_3d[:,17])
        # x19= self.layer_19(input_3d[:,18])
        # x20= self.layer_20(input_3d[:,19])
        # x21= self.layer_21(input_3d[:,20])
        # x22= self.layer_22(input_3d[:,21])
        # x23= self.layer_23(input_3d[:,22])
        # x24= self.layer_24(input_3d[:,23])
        out = torch.cat([x1, x2, x3, x4, x5,x6,x7], dim=1)
        return out

class Pos2dDiscriminator(nn.Module):
    def __init__(self,num_joints=24,channel=1000,channel_mid=100):
        super(Pos2dDiscriminator,self).__init__()

        self.layer_1 = nn.Linear(24*2, channel)
        self.layer_2 = nn.Linear(channel, channel)
        self.layer_3 = nn.Linear(channel, channel)
        self.layer_last = nn.Linear(channel, channel_mid)
        self.layer_pred = nn.Linear(channel_mid, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, input_2d):
        input_2d=input_2d.reshape(-1,24*2)
        x=self.relu(self.layer_1(input_2d))
        x=self.relu(self.layer_2(x))
        x=self.relu(self.layer_3(x))
        x=self.relu(self.layer_last(x))
        out=self.layer_pred(x)
        return out

def model_preparation(args):

    # Create model: G and D
    print("==> Creating model...")
    device = torch.device("cuda")

    # generator for PoseAug
    model_G = PoseGenerator(args).to(device)
    model_G.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_G.parameters()) / 1000000.0))

    # discriminator for 3D
    model_d3d = Pos3dDiscriminator().to(device)
    model_d3d.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d3d.parameters()) / 1000000.0))

    # discriminator for 2D
    model_d2d = Pos2dDiscriminator().to(device)
    model_d2d.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d2d.parameters()) / 1000000.0))

    # SPIN 3D pose estimator
    SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'
    model_spin = hmr(SMPL_MEAN_PARAMS)
    checkpoint = torch.load('data/data/model_checkpoint.pt') #'data/data/model_checkpoint.pt' 'models/checkpoint4.pth'
    model_spin.load_state_dict(checkpoint['model'], strict=False) #'model' 'model_state_dict'

    # prepare optimizer
    g_optimizer = torch.optim.Adam(model_G.parameters(), lr=args.lr_g)
    d3d_optimizer = torch.optim.Adam(model_d3d.parameters(), lr=args.lr_d)
    d2d_optimizer = torch.optim.Adam(model_d2d.parameters(), lr=args.lr_d)
    spin_optimizer = torch.optim.Adam(model_spin.parameters(), lr=args.lr_spin, weight_decay=0)
    # prepare scheduler
    g_lr_scheduler = get_scheduler(g_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d3d_lr_scheduler = get_scheduler(d3d_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d2d_lr_scheduler = get_scheduler(d2d_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
 
    return {
        'model_G': model_G,
        'model_d3d': model_d3d,
        'model_d2d': model_d2d,
        'model_spin': model_spin,
        'optimizer_G': g_optimizer,
        'optimizer_d3d': d3d_optimizer,
        'optimizer_d2d': d2d_optimizer,
        'optimizer_spin': spin_optimizer,
        'scheduler_G': g_lr_scheduler,
        'scheduler_d3d': d3d_lr_scheduler,
        'scheduler_d2d': d2d_lr_scheduler
    }

from torch.autograd import Variable

def get_discriminator_accuracy(prediction, label):
    '''
    this is to get discriminator accuracy for tensorboard
    input is tensor -> convert to numpy
    :param tensor_in: Bs x Score :: where score > 0.5 mean True.
    :return:
    '''
    # get numpy from tensor
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    rlt = np.abs(prediction - label)
    rlt = np.where(rlt > 0.5, 0, 1)
    num_of_correct = np.sum(rlt)
    accuracy = num_of_correct / label.shape[0]
    return accuracy

def get_adv_loss(model_dis, data_real, data_fake, criterion, summary, writer, writer_name):
    device = torch.device("cuda")
    # Adversarial losses
    real_3d = model_dis(data_real)
    fake_3d = model_dis(data_fake)

    real_label_3d = Variable(torch.ones(real_3d.size())).to(device)
    fake_label_3d = Variable(torch.zeros(fake_3d.size())).to(device)

    # adv loss
    # adv_3d_loss = criterion(real_3d, fake_3d)    # choice either one

    adv_3d_real_loss = criterion(real_3d, fake_label_3d)
    adv_3d_fake_loss = criterion(fake_3d, real_label_3d)
    # Total discriminators losses
    # adv_3d_loss = (adv_3d_real_loss + adv_3d_fake_loss) * 0.5
    adv_3d_loss =  adv_3d_fake_loss* 0.5
    # monitor training process
    ###################################################
    real_acc = get_discriminator_accuracy(real_3d.reshape(-1), real_label_3d.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_3d.reshape(-1), fake_label_3d.reshape(-1))
    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_adv_loss'.format(writer_name), adv_3d_loss.item(),
                      summary.train_iter_num)
    return adv_3d_loss


def train_dis(model_dis, data_real, data_fake, criterion, summary, writer, writer_name, fake_data_pool, optimizer):
    device = torch.device("cuda")
    optimizer.zero_grad()

    data_real = data_real.clone().detach().to(device)
    data_fake = data_fake.clone().detach().to(device)
    # store the fake buffer for discriminator training.
    data_fake = Variable(torch.Tensor(np.asarray(fake_data_pool(np.asarray(data_fake.cpu().detach()))))).to(device)

    # predicte the label
    real_pre = model_dis(data_real)
    fake_pre = model_dis(data_fake)

    real_label = Variable(torch.ones(real_pre.size())).to(device)
    fake_label = Variable(torch.zeros(fake_pre.size())).to(device)
    dis_real_loss = criterion(real_pre, real_label)
    dis_fake_loss = criterion(fake_pre, fake_label)

    # Total discriminators losses
    dis_loss = (dis_real_loss + dis_fake_loss) * 0.5

    # record acc
    real_acc = get_discriminator_accuracy(real_pre.reshape(-1), real_label.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_pre.reshape(-1), fake_label.reshape(-1))

    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_dis_loss'.format(writer_name), dis_loss.item(), summary.train_iter_num)

    # Update generators
    ###################################################
    dis_loss.backward()
    nn.utils.clip_grad_norm_(model_dis.parameters(), max_norm=1)
    optimizer.step()
    return real_acc, fake_acc

############ Model ###############################
##################################################
import math
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


"""
To use adaptator, we will try two kinds of schemes.
"""
def gn_helper(planes):
    if 0:
        return nn.BatchNorm2d(planes)
    else:
        return nn.GroupNorm(32 // 8, planes)

class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam

def hmr(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model

def create_model(ema=False):
    SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'
    model = hmr(SMPL_MEAN_PARAMS)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

######################################
######################################
## Losses
def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, alpha=0.15, needpck=False, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)

    error_pck = []
    for kp1, kp2 in zip(S1, S2):
        kp_diff_pa = np.linalg.norm(kp1 - kp2, axis=1)
        pa_pck = np.mean(kp_diff_pa < alpha)
    error_pck.append(pa_pck)
    error_pck = np.stack(error_pck)

    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    if needpck:
        return re, error_pck, S1_hat
    else:
        return re

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

##################################
##################################

# Load SMPL model
#######################
import torch
import numpy as np
# import smplx.smplx
from smplx import SMPL as _SMPL
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints
from core.utils.constants import *

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
smpl_neutral = SMPL(SMPL_MODEL_DIR,
                    create_transl=False).to(device)
# Regressor for H36m joints
J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
J_regressor_extra = torch.from_numpy(np.load(JOINT_REGRESSOR_TRAIN_EXTRA)).float()
smpl_neutral = SMPL(SMPL_MODEL_DIR, create_transl=False).to(device)
smpl_male = SMPL(SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
##################################################
##################################################
def decode_smpl_params(rotmats, betas, cam, neutral=True, pose2rot=False):
    if neutral:
        smpl_out = smpl_neutral(betas=betas, body_pose=rotmats[:,1:], global_orient=rotmats[:,0].unsqueeze(1), pose2rot=pose2rot)
    return {'s3d': smpl_out.joints, 'vts': smpl_out.vertices}

import random
def seed_everything(seed=42):
    """ we need set seed to ensure that all model has same initialization
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    print('---> seed has been set')

from core.PW3D import PW3D
from tqdm import tqdm
class evaluate():
    def __init__(self,model=None):
        self.device = torch.device('cuda')
        seed_everything(22)


        if model is None:
            SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'
            self.model = hmr(SMPL_MEAN_PARAMS)
            checkpoint = torch.load('data/data/model_checkpoint.pt')
            self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.model=model

        SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'
        self.model2 = hmr(SMPL_MEAN_PARAMS)
        checkpoint2 = torch.load('models/checkpoint_normal5.pth')
        self.model2.load_state_dict(checkpoint2['model_state_dict'], strict=False)

        self.pw3d_dataset = PW3D('3dpw')
        self.pw3d_dataloader = DataLoader(self.pw3d_dataset, batch_size=32, shuffle=False,num_workers=32)

    def inference(self):
        mpjpe, pampjpe = [], []
        uposed_mesh_error, posed_mesh_error = [],[]
        self.history_info = {}
        for step, pw3d_batch in tqdm(enumerate(self.pw3d_dataloader), total=len(self.pw3d_dataloader)):
            self.global_step = step
            self.model=self.model.to(self.device)
            self.model2=self.model2.to(self.device)
            pw3d_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in pw3d_batch.items()}

            # test using the adapted model
            eval_res = self.test(pw3d_batch)
            mpjpe.append(eval_res['mpjpe'])
            pampjpe.append(eval_res['pa-mpjpe'])
            uposed_mesh_error.append(eval_res['ume'])
            posed_mesh_error.append(eval_res['pme'])

            if self.global_step % 200 == 0:
                print(f'step:{self.global_step} \t MPJPE:{np.mean(mpjpe)*1000} \t PAMPJPE:{np.mean(pampjpe)*1000}')
        
        # save results
        mpjpe = np.mean(np.concatenate(mpjpe))
        pampjpe = np.mean(np.concatenate(pampjpe))
        uposed_mesh_error = np.mean(np.concatenate(uposed_mesh_error))
        posed_mesh_error = np.mean(np.concatenate(posed_mesh_error))
        # np.save(osp.join(self.exppath, 'mpjpe'), mpjpe)
        # np.save(osp.join(self.exppath, 'pampjpe'), pampjpe)
        # np.save(osp.join(self.exppath, 'ume'), uposed_mesh_error)
        # np.save(osp.join(self.exppath, 'pme'), posed_mesh_error)      
        print("== Final Results ==")
        print('MPJPE:', mpjpe*1000)
        print('PAMPJPE:', pampjpe*1000)
        print('Mesh Error:', uposed_mesh_error, posed_mesh_error)
        # with open(osp.join(self.exppath, 'performance.txt'), 'w') as f:
        #     _res = f'MPJPE:{mpjpe*1000}, PAMPJPE:{pampjpe*1000}, ume:{uposed_mesh_error}, pme:{posed_mesh_error}'
        #     f.write(_res)


    def test(self, databatch):
        
        gt_pose = databatch['pose']
        gt_betas = databatch['betas']
        gender = databatch['gender']            
        
        with torch.no_grad():
            # forward
            self.model.eval()
            self.model2.eval()
            images = databatch['image']
            # print(images.shape)
            pred_rotmat, pred_betas, pred_cam = self.model(images)
            pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']

            # # mohsen: added for plots
            # pred_rotmat2, pred_betas2, pred_cam2 = self.model2(images)
            # pred_smpl_out2 = decode_smpl_params(pred_rotmat2, pred_betas2, pred_cam2, neutral=True)
            # pred_vts2 = pred_smpl_out2['vts']


            # calculate metrics 
            J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to(self.device)

            gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
            gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
            gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, H36M_TO_J14, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # get unposed mesh
            t_rotmat = torch.eye(3,3).unsqueeze(0).unsqueeze(0).repeat(pred_rotmat.shape[0], pred_rotmat.shape[1], 1, 1).to(self.device)
            pred_smpl_out = decode_smpl_params(t_rotmat, pred_betas, pred_cam, neutral=True)
            unposed_pred_vts = pred_smpl_out['vts']
            unposed_gt_vertices = smpl_male(global_orient=t_rotmat[:,1:], body_pose=t_rotmat[:,0].unsqueeze(1), betas=gt_betas, pose2rot=False).vertices 
            unposed_gt_vertices_female = smpl_female(global_orient=t_rotmat[:,1:], body_pose=t_rotmat[:,0].unsqueeze(1), betas=gt_betas, pose2rot=False).vertices 
            unposed_gt_vertices[gender==1, :, :] = unposed_gt_vertices_female[gender==1, :, :]
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vts)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, H36M_TO_J14, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # # mohsen: added for plots
            # pred_keypoints_3d2 = torch.matmul(J_regressor_batch, pred_vts2)
            # pred_pelvis2 = pred_keypoints_3d2[:, [0], :].clone()
            # pred_keypoints_3d2 = pred_keypoints_3d2[:, H36M_TO_J14, :]
            # pred_keypoints_3d2 = pred_keypoints_3d2 - pred_pelvis2
            
            # ## mohsen: plot for debug
            # import matplotlib.pyplot as plt
            # # images0=databatch['raw_image']
            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # frame=10
            # ax.imshow(images[frame].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2,2,projection='3d')


            # bones = [[0,1],[1,2],[2,12],[3,12],[3,4],[4,5],[6,7],[7,8],[8,12],[9,12],[9,10],[10,11],[12,13]]
            # # bones_r=[[0,2],[2,4],[0,6],[6,9],[9,11],[11,13]]
            # # spin=[[0,6],[1,6],[6,7],[6,8]]
            # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in bones:
            #     col='b'
            #     ax.plot(gt_keypoints_3d[frame,limb,0].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,1].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # # plt.show()
            # # ax = fig.add_subplot(1,3,3,projection='3d')
            # for limb in bones:
            #     col='g'
            #     ax.plot(pred_keypoints_3d2[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d2[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d2[frame,limb,2].detach().cpu().numpy(),col)
            #     ax.set_box_aspect((np.ptp(pred_keypoints_3d2[frame,:,0].detach().cpu().numpy()), np.ptp(pred_keypoints_3d2[frame,:,1].detach().cpu().numpy()), np.ptp(pred_keypoints_3d2[frame,:,2].detach().cpu().numpy())))

            # for limb in bones:
            #     col='r'
            #     ax.plot(pred_keypoints_3d[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # plt.show()

            # 1. MPJPE
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            # 2. PA-MPJPE
            r_error, pck_error,_ = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),needpck=True, reduction=None)
            results = {'mpjpe': error, 'pa-mpjpe': r_error, 'pck': pck_error}
            # 3. shape evaluation
            unposed_mesh_error = torch.sqrt(((unposed_gt_vertices - unposed_pred_vts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            posed_mesh_error = torch.sqrt(((gt_vertices - pred_vts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            results['ume'] = unposed_mesh_error
            results['pme'] = posed_mesh_error
        return results

class pose_dataset(Dataset):
    def __init__(self,annot_dir,image_dir):
        self.image_dir=image_dir
        self.annot=np.load(annot_dir)
        
    def __len__(self):
        return 9000 #len(self.annot)
    
    def __getitem__(self,idx):
        #mohsen: change this to +1000 for 20221004 generated samples:
        idx_=idx #+1000     
        image = read_image(self.image_dir+f'{idx_:05d}.png')
        image = image[120:392,120:392,:]
        image = process_sample(image.copy())
        image = resize(image, (3,224, 224),
                        anti_aliasing=True)
        pose=self.annot[idx]

        sample={'image':image,'pose':pose}

        return sample

class mpii_nerf_dataset(Dataset):

    def __init__(self,annot_dir_mpii,image_dir_mpii,annot_dir_nerf,image_dir_nerf):
        self.frac=10
        self.image_dir_mpii=image_dir_mpii
        self.annot_mpii=np.load(annot_dir_mpii)
        self.pose=self.annot_mpii['pose']
        self.imgname=self.annot_mpii['imgname']
        self.center=self.annot_mpii['center']
        self.scale=self.annot_mpii['scale']  

        self.image_dir_nerf=image_dir_nerf
        self.annot_nerf=np.load(annot_dir_nerf)[:1000]
    
    def __len__(self):
        return round(len(self.annot_nerf)*self.frac/(self.frac-1)-10)

    def __getitem__(self,idx):

        if idx%self.frac==0:
            idx_mpii=idx//self.frac
            image = read_image(self.image_dir_mpii+self.imgname[idx_mpii])
            center=self.center[idx_mpii]
            scale=self.scale[idx_mpii]*200
            xy1=center-scale/2
            xy2=center+scale/2
            xy1[0],xy2[0]=np.clip(xy1[0],0,image.shape[0]),np.clip(xy2[0],0,image.shape[1])
            xy1[1],xy2[1]=np.clip(xy1[1],0,image.shape[1]),np.clip(xy2[1],0,image.shape[0])
            image=image[int(xy1[1]):int(xy2[1]),int(xy1[0]):int(xy2[0])]
            image = process_sample(image.copy())
            image = resize(image, (3,224, 224),
                            anti_aliasing=True)
            pose=np.reshape(self.pose[idx_mpii],(24,3))
            pose=get_smpl_l2ws(pose,scale=0.4)[:,:3,-1]
            sample={'image':image,'pose':pose}

        else: # idx%2==1:
            idx_nerf=idx-idx//self.frac-1
            idx_nerf_=idx_nerf+1000

            image = read_image(self.image_dir_nerf+f'{idx_nerf_:05d}.png')
            image = image[120:392,120:392,:]
            image = process_sample(image.copy())
            image = resize(image, (3,224, 224),
                            anti_aliasing=True)
            pose=self.annot_nerf[idx_nerf]

            sample={'image':image,'pose':pose}
        
        return sample

class mpii_dataset(Dataset):
    def __init__(self,annot_dir,image_dir):
        self.image_dir=image_dir
        self.annot=np.load(annot_dir)
        self.pose=self.annot['pose']
        self.imgname=self.annot['imgname']
        self.center=self.annot['center']
        self.scale=self.annot['scale']

    def __len__(self):
        return len(self.pose)
    
    def __getitem__(self,idx): 
        image = read_image(self.image_dir+self.imgname[idx])
        center=self.center[idx]
        scale=self.scale[idx]*200
        xy1=center-scale/2
        xy2=center+scale/2
        xy1[0],xy2[0]=np.clip(xy1[0],0,image.shape[0]),np.clip(xy2[0],0,image.shape[1])
        xy1[1],xy2[1]=np.clip(xy1[1],0,image.shape[1]),np.clip(xy2[1],0,image.shape[0])
        image=image[int(xy1[1]):int(xy2[1]),int(xy1[0]):int(xy2[0])]
        image = process_sample(image.copy())
        image = resize(image, (3,224, 224),
                        anti_aliasing=True)
        pose=np.reshape(self.pose[idx],(24,3))
        sample={'image':image,'pose':pose}

        return sample


def get_one_box(det_output, thrd=0.9):
    max_area = 0
    max_bbox = None

    if det_output['boxes'].shape[0] == 0 or thrd < 1e-5:
        return None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(area)
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox

class agora_dataset(Dataset):
    def __init__(self,image_dir,pose_dir):
        self.image_dir=image_dir
        self.image_names=glob.glob(self.image_dir)
        with open(pose_dir, 'rb') as f:
            self.pose = pickle.load(f)
        
        self.det_model = fasterrcnn_resnet50_fpn(pretrained=True).to('cuda')
        self.det_model.eval()
    def __len__(self):
        return len(self.pose)
    
    def __getitem__(self,idx): 
        image_name=self.pose[idx]['image_name']
        
        image = read_image('/media/ExtHDD/Mohsen_data/AGORA/test/'+image_name)
            
        pose=np.asarray(self.pose[idx]['2dpose'])[0]
        # x1,y1=max(0,int(min(pose[:,0]))-50),max(0,int(min(pose[:,1]))-50)
        # x2,y2=min(3840,int(max(pose[:,0]))+50),min(2160,int(max(pose[:,1]))+50)
        # center_x,center_y=(pose[11]+pose[12])/2
        # # x1,y1=max(0,int(min(pose[:,0]))-100),max(0,int(min(pose[:,1]))-100)
        # # x2,y2=min(3840,int(max(pose[:,0]))+100),min(2160,int(max(pose[:,1]))+100)
        # wh=max(y2-y1,x2-x1)/2
        # x1,y1,x2,y2=max(0,int(center_x-wh)),max(0,int(center_y-wh)),min(3840,int(center_x+wh)),min(2160,int(center_y+wh))
        # image=image[y1:y2,x1:x2,:]

        # # print(image_name,torch.tensor(image).shape,x1,x2,y1,y2)
        # # det_input = det_transform(image).to('cuda')
        # # det_output = self.det_model([det_input])[0]
        # # tight_bbox = get_one_box(det_output)  # xyxy
        


        # image = process_sample(image.copy())

        # image = resize(image, (3,224, 224),
        #                 anti_aliasing=True)

        _,image=process_image(img_file='/media/ExtHDD/Mohsen_data/AGORA/test/'+image_name,openpose_file=pose)
        
        sample={'image':image[0],'pose2d':torch.tensor(pose),'image_name':image_name}

        return sample


class ski_dataset(Dataset):
    def __init__(self,image_dir,split='test'):
        self.split=split
        self._labels= h5py.File('/media/ExtHDD/Mohsen_data/SKI/'+split+'/labels.h5', 'r')

    def __len__(self):
        return len(self._labels['seq'])
    
    def __getitem__(self,idx): 
        seq   = int(self._labels['seq'][idx])
        cam   = int(self._labels['cam'][idx])
        frame = int(self._labels['frame'][idx])
        img_path = '/media/ExtHDD/Mohsen_data/SKI/'+self.split+'/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq,cam,frame)
        
        

        gt_pose=self._labels['3D'][idx].reshape([-1,3])[[4,1,5,2,6,3,8,10,11,14,12,15,13,16],:]
        pose_2d = self._labels['2D'][idx].reshape([1,-1,2]) # in range 0..1
        
        # get image id
        img_id = int(frame)

        # load ground truth, including bbox, keypoints, image size
        # label = copy.deepcopy(self._labels[idx])
        img = read_image(img_path)


        image = process_sample(img.copy())

        image = resize(image, (3,224, 224),
                        anti_aliasing=True)
        # print(pose_2d.shape)
        # _,image=process_image(img_file=img_path,openpose_file=pose_2d[0])
        sample={'image':torch.tensor(image),'pose_3d':torch.tensor(gt_pose)}

        return sample


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width : int or float
        Boundary width.
    height : int or float
        Boundary height.

    Returns
    -------
    type
        Description of returned object.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))

def cam2pixel_matrix(cam_coord, intrinsic_param):
    cam_coord = cam_coord.transpose(1, 0)
    cam_homogeneous_coord = np.concatenate((cam_coord, np.ones((1, cam_coord.shape[1]), dtype=np.float32)), axis=0)
    img_coord = np.dot(intrinsic_param, cam_homogeneous_coord) / (cam_coord[2, :] + 1e-8)
    img_coord = np.concatenate((img_coord[:2, :], cam_coord[2:3, :]), axis=0)
    return img_coord.transpose(1, 0)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = '/media/ExtHDD/Mohsen_data//mpi_inf_3dhp'
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        self.data = np.load('/home/mgholami/A-NeRF/data/spin_data/data/dataset_extras/mpi_inf_3dhp_valid.npz')
        self.imgname = self.data['imgname']
        
        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [IMG_RES, IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [IMG_RES, IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)

class HP3D(Dataset):
    """ MPI-INF-3DHP dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/3dhp'
        Path to the 3dhp dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [i - 1 for i in [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]]
    EVAL_JOINTS_17 = [
        14,
        11, 12, 13,
        8, 9, 10,
        15, 1,
        16, 0,
        5, 6, 7,
        2, 3, 4
    ]
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    # EVAL_JOINTS = [10, 8, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6, 0, 7, 9]  # h36m -> 3dhp

    # num_joints = 28
    joints_name = ('spine3', 'spine4', 'spine2', 'spine', 'pelvis',                         # 4
                   'neck', 'head', 'head_top',                                              # 7
                   'left_clavicle', 'left_shoulder', 'left_elbow',                          # 10
                   'left_wrist', 'left_hand', 'right_clavicle',                             # 13
                   'right_shoulder', 'right_elbow', 'right_wrist',                          # 16
                   'right_hand', 'left_hip', 'left_knee',                                   # 19
                   'left_ankle', 'left_foot', 'left_toe',                                   # 22
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe')     # 27
    skeleton = ((0, 2), (1, 0), (2, 3), (3, 4),
                (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                (17, 16), (18, 4), (19, 18), (20, 19), (21, 20), (22, 21),
                (23, 4), (24, 23), (25, 24), (26, 25), (27, 26)
                )
    skeleton = (
        (1, 0), (2, 1), (3, 2),         # 2
        (4, 0), (5, 4), (6, 5),         # 5
        (7, 0), (8, 7),                 # 7
        (9, 8), (10, 9),                # 9
        (11, 7), (12, 11), (13, 12),    # 12
        (14, 7), (15, 14), (16, 15),    # 15
    )
    mean_bone_len = None
    test_seqs = (1, 2, 3, 4, 5, 6)
    joint_groups = {'Head': [0], 'Neck': [1], 'Shou': [2, 5], 'Elbow': [3, 6], 'Wrist': [4, 7], 'Hip': [8, 11], 'Knee': [9, 12], 'Ankle': [10, 13]}
    # activity_name full name: ('Standing/Walking','Exercising','Sitting','Reaching/Crouching','On The Floor','Sports','Miscellaneous')
    activity_name = ('Stand', 'Exe', 'Sit', 'Reach', 'Floor', 'Sports', 'Miscell')
    pck_thres = 150
    auc_thres = list(range(0, 155, 5))

    def __init__(self,
                 ann_file,
                 root='./data/3dhp',
                 train=False,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):
        self._cfg = cfg

        self._ann_file = os.path.join(
            root, f'annotation_mpi_inf_3dhp_{ann_file}.json')
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        self._scale_factor = 0.3
        self._color_factor = 0.2
        self._rot = 30
        self._input_size = [256,256]
        self._output_size = [64,64]

        self._occlusion = True

        self._crop = 'padding'
        self._sigma = 2
        self._depth_dim = 64

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self.num_joints = 28 if self._train else 17

        self.num_joints_half_body = 8
        self.prob_half_body = -1

        self.augment = 'none'

        self._loss_type = 'L1LossDimSMPLCam'
        self.kinematic = False

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.root_idx = self.joints_name.index('pelvis') if self._train else self.EVAL_JOINTS.index(self.joints_name.index('pelvis'))

        self.transformation = SimpleTransform3DSMPLCam(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=False,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg,
            loss_type=self._loss_type, two_d=True,
                root_idx=self.root_idx)

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


        return img, label

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)
        # iterate through the annotations
        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(ann['bbox']), width, height)

            intrinsic_param = np.array(ann['cam_param']['intrinsic_param'], dtype=np.float32)

            f = np.array([intrinsic_param[0, 0], intrinsic_param[1, 1]], dtype=np.float32)
            c = np.array([intrinsic_param[0, 2], intrinsic_param[1, 2]], dtype=np.float32)

            joint_cam = np.array(ann['keypoints_cam'])

            joint_img = cam2pixel_matrix(joint_cam, intrinsic_param)
            joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]
            joint_vis = np.ones((self.num_joints, 3))

            root_cam = joint_cam[self.root_idx]

            abs_path = os.path.join(self._root, 'mpi_inf_3dhp_{}_set'.format('train' if self._train else 'test'), ann['file_name'])

            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'img_name': ann['file_name'],
                'width': width,
                'height': height,
                'joint_img': joint_img,
                'joint_vis': joint_vis,
                'joint_cam': joint_cam,
                'root_cam': root_cam,
                'intrinsic_param': intrinsic_param,
                'f': f,
                'c': c
            })
            if not self._train:
                labels[-1]['activity_id'] = ann['activity_id']
        return items, labels

def evaluate_ski(model,model2=None):

    if model2!=None:
        model_spin2=model2.to('cuda')
        model_spin2.eval()         
    model_spin= model.to('cuda')
    model_spin.eval()
    
    image_dir='/media/ExtHDD/Mohsen_data/ski/*'
    

    dataset_ski=ski_dataset(image_dir)
    data_loader_ski=DataLoader(dataset_ski,batch_size=32,num_workers=8)
    mpjp_error=0
    pampjp_error=0
    pck_error=0
    len_tot=0

    for idx, sample in enumerate(data_loader_ski):
        image,gt_keypoints_3d=sample['image'].to('cuda'),sample['pose_3d'].to('cuda')
        with torch.no_grad():
            # forward
            
            # print(images.shape)
            pred_rotmat, pred_betas, pred_cam = model_spin(image)
            pred_keypoints_3d=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            
            if model2!=None:
                pred_rotmat2, pred_betas, pred_cam = model_spin2(image)
                pred_keypoints_3d2=get_smpl_l2ws_torch(pred_rotmat2,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]

            pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']
            
            if model2!=None:
                pred_smpl_out2 = decode_smpl_params(pred_rotmat2, pred_betas, pred_cam, neutral=True)
                pred_vts2 = pred_smpl_out2['vts']

            # # calculate metrics 
            J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to('cuda')
            J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to('cuda')

            # gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            # gt_keypoints_3d = gt_keypoints_3d
            # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            EVAL_JOINTS = [1,4,2,5,3,6,8,10,11,14,12,15,13,16]
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vts)
            if model2!=None:
                # # Get 14 predicted joints from the mesh
                pred_keypoints_3d2 = torch.matmul(J_regressor_batch, pred_vts2)
            
            
            EVAL_JOINTS24to14 = [1,2,4,5,7,8,12,15,16,17,18,19,20,21]
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, EVAL_JOINTS, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            if model2!=None:
                pred_pelvis2 = pred_keypoints_3d2[:, [0], :].clone()
                pred_keypoints_3d2 = pred_keypoints_3d2[:, EVAL_JOINTS, :]
                pred_keypoints_3d2 = pred_keypoints_3d2 - pred_pelvis2
            
            scale_pred=torch.linalg.norm(pred_keypoints_3d[:,6:7]-pred_keypoints_3d[:,:1],dim=-1,keepdims=True)
            scale_gt=torch.linalg.norm(gt_keypoints_3d[:,6:7]-gt_keypoints_3d[:,:1],dim=-1,keepdims=True)
            # pred_keypoints_3d=pred_keypoints_3d*scale_gt/scale_pred
            # 1. MPJPE
            mpjp_error += torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean().cpu().numpy()*len(pred_keypoints_3d)
            # 2. PA-MPJPE
            r_e, pck_e,_ = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),needpck=True, reduction=None)
            pampjp_error += r_e.mean()*len(gt_keypoints_3d)
            pck_error += pck_e*len(gt_keypoints_3d)
            len_tot += len(gt_keypoints_3d)
            results = {'mpjpe': mpjp_error/len_tot, 'pa-mpjpe':pampjp_error/len_tot, 'pck': pck_error/len_tot}
    
            # ## mohsen: plot for debugging
            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # frame=20
            # ax.imshow(image[frame].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2,2,projection='3d')
            # # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # # # pose_numpy=pred_xyz_jts_24.cpu().detach().numpy()
            # # # # pose_numpy=pose_numpy-pose_numpy[:,:1]       
            # # # for limb in limbs24:
            # # #     ax.plot(pose_numpy[frame,limb,0],pose_numpy[frame,limb,1],pose_numpy[frame,limb,2],'r')
            # # # # plt.show()
            # # # print(gt_pose.shape)
            # # # gt_pose_numpy=gt_pose.cpu().detach().numpy() 
            # # # # gt_pose_numpy=gt_pose_numpy-gt_pose_numpy[:,:1]
            
            # # # # gt_pose_numpy*=0.4 
            # bones = [[0,2],[2,4],[1,3],[3,5],[0,6],[1,6],[6,7],[6,8],[8,10],[10,12],[6,9],[9,11],[11,13]]
            # bones_r=[[0,2],[2,4],[0,6],[6,9],[9,11],[11,13]]
            # spin=[[0,6],[1,6],[6,7],[6,8]]
            # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in bones:
            #     col='b'
            #     if limb in bones_r:
            #         col='b'
            #     elif limb in spin:
            #         col='b'
            #     ax.plot(gt_keypoints_3d[frame,limb,0].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,1].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # # plt.show()
            # # ax = fig.add_subplot(1,3,3,projection='3d')

            # for limb in bones:
            #     col='r'
            #     if limb in bones_r:
            #         col='r'
            #     elif limb in spin:
            #         col='r'
            #     ax.plot(pred_keypoints_3d[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # for limb in bones:
            #     col='g'
            #     if limb in bones_r:
            #         col='g'
            #     elif limb in spin:
            #         col='g'
            #     ax.plot(pred_keypoints_3d2[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d2[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d2[frame,limb,2].detach().cpu().numpy(),col)
            # plt.show()
   
    print(results)

def train_ski(model,model2=None):

    if model2!=None:
        model_spin2=model2.to('cuda')
        model_spin2.eval()         
    model_spin= model.to('cuda')
    model_spin.train()

    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
        
    model_spin.apply(set_bn_eval)
    spin_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_spin.parameters()), lr=args.lr_spin, weight_decay=0)   

    image_dir='/media/ExtHDD/Mohsen_data/ski/*'
    

    dataset_ski=ski_dataset(image_dir,split='train2/train')
    data_loader_ski=DataLoader(dataset_ski,batch_size=32,num_workers=8,shuffle=True)

    
    for epoch in range(args.epochs):
        for idx, sample in tqdm(enumerate(data_loader_ski)):
            spin_optimizer.zero_grad()
            image,gt_keypoints_3d=sample['image'].to('cuda'),sample['pose_3d'].to('cuda')
        
            # forward 
            # print(images.shape)
            pred_rotmat, pred_betas, pred_cam = model_spin(image)
            pred_keypoints_3d=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]

            pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']


            # # # calculate metrics 
            J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to('cuda')

        
            # # # Get 14 predicted joints from the mesh
            EVAL_JOINTS = [1,4,2,5,3,6,8,10,11,14,12,15,13,16]
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vts)
            
            
            EVAL_JOINTS24to14 = [1,2,4,5,7,8,12,15,16,17,18,19,20,21]
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, EVAL_JOINTS, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            
            scale_pred=torch.linalg.norm(pred_keypoints_3d[:,6:7]-pred_keypoints_3d[:,:1],dim=-1,keepdims=True)
            scale_gt=torch.linalg.norm(gt_keypoints_3d[:,6:7]-gt_keypoints_3d[:,:1],dim=-1,keepdims=True)
            pred_keypoints_3d=pred_keypoints_3d*scale_gt/scale_pred
            # 1. MPJPE
            loss = mpjpe(pred_keypoints_3d, gt_keypoints_3d)
            loss.backward()
            spin_optimizer.step()
            
            
    
            # ## mohsen: plot for debugging
            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # frame=20
            # ax.imshow(image[frame].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2,2,projection='3d')
            # # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # # # pose_numpy=pred_xyz_jts_24.cpu().detach().numpy()
            # # # # pose_numpy=pose_numpy-pose_numpy[:,:1]       
            # # # for limb in limbs24:
            # # #     ax.plot(pose_numpy[frame,limb,0],pose_numpy[frame,limb,1],pose_numpy[frame,limb,2],'r')
            # # # # plt.show()
            # # # print(gt_pose.shape)
            # # # gt_pose_numpy=gt_pose.cpu().detach().numpy() 
            # # # # gt_pose_numpy=gt_pose_numpy-gt_pose_numpy[:,:1]
            
            # # # # gt_pose_numpy*=0.4 
            # bones = [[0,2],[2,4],[1,3],[3,5],[0,6],[1,6],[6,7],[6,8],[8,10],[10,12],[6,9],[9,11],[11,13]]
            # bones_r=[[0,2],[2,4],[0,6],[6,9],[9,11],[11,13]]
            # spin=[[0,6],[1,6],[6,7],[6,8]]
            # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in bones:
            #     col='b'
            #     if limb in bones_r:
            #         col='b'
            #     elif limb in spin:
            #         col='b'
            #     ax.plot(gt_keypoints_3d[frame,limb,0].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,1].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # # plt.show()
            # # ax = fig.add_subplot(1,3,3,projection='3d')

            # for limb in bones:
            #     col='r'
            #     if limb in bones_r:
            #         col='r'
            #     elif limb in spin:
            #         col='r'
            #     ax.plot(pred_keypoints_3d[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # for limb in bones:
            #     col='g'
            #     if limb in bones_r:
            #         col='g'
            #     elif limb in spin:
            #         col='g'
            #     ax.plot(pred_keypoints_3d2[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d2[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d2[frame,limb,2].detach().cpu().numpy(),col)
            # plt.show()
        evaluate_ski(model=model_spin.eval())
   
    return(model_spin)


def evaluate_3dhp(model):
             
    model_spin= model.to('cuda')
    model_spin.eval()
    
    dataset = BaseDataset(None, 'mpi-inf-3dhp', is_train=False)


    data_loader_3dhp=DataLoader(dataset,batch_size=32)

    mpjp_error=0
    pampjp_error=0
    pck_error=0
    len_tot=0

    for idx, sample in tqdm(enumerate(data_loader_3dhp)):
        image,gt_keypoints_3d=sample['img'].to('cuda'),sample['pose_3d'].to('cuda')
        with torch.no_grad():
            # forward
     
            # print(images.shape)
            pred_rotmat, pred_betas, pred_cam = model_spin(image)
            # pred_keypoints_3d=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)  


            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # frame=0
            # ax.imshow(image[frame].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2,2,projection='3d')
            # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # # pose_numpy=pred_xyz_jts_24.cpu().detach().numpy()
            # # # pose_numpy=pose_numpy-pose_numpy[:,:1]       
            # # for limb in limbs24:
            # #     ax.plot(pose_numpy[frame,limb,0],pose_numpy[frame,limb,1],pose_numpy[frame,limb,2],'r')
            # # # plt.show()
            # # print(gt_pose.shape)
            # # gt_pose_numpy=gt_pose.cpu().detach().numpy() 
            # # # gt_pose_numpy=gt_pose_numpy-gt_pose_numpy[:,:1]
            
            # # # gt_pose_numpy*=0.4 
            # bones = [[0,2],[2,4],[1,3],[3,5],[0,6],[1,6],[6,7],[6,8],[8,10],[10,12],[6,9],[9,11],[11,13]]
            # bones_r=[[0,2],[2,4],[0,6],[6,9],[9,11],[11,13]]
            # spin=[[0,6],[1,6],[6,7],[6,8]]
            # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in limbs24:
            #     # col='b'
            #     # if limb in bones_r:
            #     #     col='r'
            #     # elif limb in spin:
            #     #     col='g'
            #     col='b'
            #     ax.plot(gt_keypoints_3d[frame,limb,0].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,1].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # # plt.show()
            # # ax = fig.add_subplot(1,3,3,projection='3d')

            # for limb in limbs24:
            #     # col='b'
            #     # if limb in bones_r:
            #     #     col='r'
            #     # elif limb in spin:
            #     #     col='g'
            #     col='r'
            #     ax.plot(pred_keypoints_3d[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # plt.show()

            # pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            # pred_vts = pred_smpl_out['vts']

            # # calculate metrics 
            # J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to('cuda')

            # gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            # gt_keypoints_3d = gt_keypoints_3d
            # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
          
            # # Get 14 predicted joints from the mesh
            # EVAL_JOINTS = [1,4,2,5,3,6,8,10,11,14,12,15,13,16]
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            
            gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J17, :-1]  

            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, H36M_TO_J17 , :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            # 1. MPJPE
            mpjp_error += torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean().cpu().numpy()*len(pred_keypoints_3d)
            # 2. PA-MPJPE
            r_e, pck_e,_ = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),needpck=True, reduction=None)
            pampjp_error += r_e.mean()*len(gt_keypoints_3d)
            pck_error += pck_e*len(gt_keypoints_3d)
            len_tot += len(gt_keypoints_3d)
            results = {'mpjpe': mpjp_error/len_tot, 'pa-mpjpe':pampjp_error/len_tot, 'pck': pck_error/len_tot}
    
            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # frame=0
            # ax.imshow(image[frame].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2,2,projection='3d')
            # # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # # pose_numpy=pred_xyz_jts_24.cpu().detach().numpy()
            # # # pose_numpy=pose_numpy-pose_numpy[:,:1]       
            # # for limb in limbs24:
            # #     ax.plot(pose_numpy[frame,limb,0],pose_numpy[frame,limb,1],pose_numpy[frame,limb,2],'r')
            # # # plt.show()
            # # print(gt_pose.shape)
            # # gt_pose_numpy=gt_pose.cpu().detach().numpy() 
            # # # gt_pose_numpy=gt_pose_numpy-gt_pose_numpy[:,:1]
            
            # # # gt_pose_numpy*=0.4 
            # bones = [[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[14,11],[11,13]]
            # bones_r=[[0,2],[2,4],[0,6],[6,9],[9,11],[11,13]]
            # spin=[[0,6],[1,6],[6,7],[6,8]]
            # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in bones:
            #     # col='b'
            #     # if limb in bones_r:
            #     #     col='r'
            #     # elif limb in spin:
            #     #     col='g'
            #     col='b'
            #     ax.plot(gt_keypoints_3d[frame,limb,0].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,1].detach().cpu().numpy(),gt_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # # plt.show()
            # # ax = fig.add_subplot(1,3,3,projection='3d')

            # for limb in bones:
            #     # col='b'
            #     # if limb in bones_r:
            #     #     col='r'
            #     # elif limb in spin:
            #     #     col='g'
            #     col='r'
            #     ax.plot(pred_keypoints_3d[frame,limb,0].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,1].detach().cpu().numpy(),pred_keypoints_3d[frame,limb,2].detach().cpu().numpy(),col)
            # plt.show()

            
   
            # print(results)   
    print(results)  

def evaluate_agora(model):
    args=config_parser().parse_args()
    # load data
    # data_dict = data_preparation(args)
    # model_dict=model_preparation(args)

    model_spin= model.to('cuda')
    model_spin.eval()
    
    image_dir='/media/ExtHDD/Mohsen_data/AGORA/test/*'
    pose_dir='/home/mgholami/HRNet/new_test_agora.pkl' 

    dataset_agora=agora_dataset(image_dir,pose_dir)
    data_loader_agora=DataLoader(dataset_agora,batch_size=1,num_workers=8)
    
    for idx, sample in enumerate(data_loader_agora):
        image,pose2d,image_name=sample['image'].to('cuda'),sample['pose2d'].to('cuda'),sample['image_name']
        # print(image.shape)
        with torch.no_grad():
            model_spin.eval()
            pred_rotmat, _, _ = model_spin(image)
            pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            pose=pose-pose[:,:1]
            

            # ## plot for debugging
            # #####################
            # frame=0
            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # ax.imshow(image[frame].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2, 2,projection='3d')
            # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in limbs24:
            #     pose_numpy=pose.cpu().detach().numpy()
            #     ax.plot(pose_numpy[frame,limb,0],pose_numpy[frame,limb,1],pose_numpy[frame,limb,2],'r')

            # ax.set_box_aspect((np.ptp(pose_numpy[frame,:,0]), np.ptp(pose_numpy[frame,:,1]), np.ptp(pose_numpy[frame,:,2]))) 
            # plt.show()                  
            # #####################
            
            pred_rotmat, pred_betas, pred_cam = model_spin(image)
            pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']
            # pred_pose3d=pred_smpl_out['s3d'][0,-24:]
            pred_pose3d=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            # joint_reorder=[]
            # pred_pose3d=[]

            # # calculate metrics 
            # J_regressor_batch = J_regressor_extra[None, :].expand(pred_vts.shape[0], -1, -1).to('cuda')

            # pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vts)

            
            pred_pose2d=pred_pose3d[0,:,:2]
            
            root=(pose2d[0,11].clone()+pose2d[0,12].clone())/2
            pred_pose2d=pred_pose2d-pred_pose2d[:1]
            pose2d=pose2d[0]-root
            scale_pred=torch.linalg.norm(pred_pose2d)
            scale_hrnet=torch.linalg.norm(pose2d)
            pred_pose2d=pred_pose2d/scale_pred*scale_hrnet
            pred_pose2d=pred_pose2d+root
            pose2d+=root
            # print(pred_pose2d.size(),pred_vts[0].size(),pred_pose3d[0].size())
            output={}
            output['joints']=pred_pose2d.cpu().numpy()
            output['verts']=pred_vts[0].cpu().numpy()
            output['allSmplJoints3d']=pred_pose3d[0].cpu().numpy()
            count=0
            
            while os.path.exists('predictions/'+image_name[0][:-4]+'_personId_'+str(count)+'.pkl'):
                count+=1
            with open('predictions/'+image_name[0][:-4]+'_personId_'+str(count)+'.pkl','wb') as f:
                pickle.dump(output,f,pickle.HIGHEST_PROTOCOL)

            # ### plot for debugging
            # ######################
            # fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)
            # ax.imshow(image[0].permute(1,2,0).cpu().detach().numpy())
            # ax = fig.add_subplot(1,2, 2)
            # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            # for limb in limbs24:
            #     pose_numpy=pred_pose2d.cpu().detach().numpy()
            #     ax.plot(pose_numpy[limb,0],pose_numpy[limb,1],'r')
            # # limbs17_hrnet = [[1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]
            # # for limb in limbs17_hrnet:
            # #     pose_numpy=pose2d.cpu().detach().numpy()
            # #     ax.plot(pose_numpy[limb,0],pose_numpy[limb,1],'b')

            # # # ax.set_box_aspect((np.ptp(pose_numpy[:,0]), np.ptp(pose_numpy[:,1])))
            # plt.show()   
            # ###########################          
            

def train_spin(model,evalset=['3dpw',],trainset=['nerf',]):
    args=config_parser().parse_args()
    # load data
    # data_dict = data_preparation(args)
    # model_dict=model_preparation(args)

    model_spin= model.to('cuda')
    
    # model_spin.train()
    # torch.set_grad_enabled(True)
    # print(model_spin)
    # for name ,child in (model_spin.named_children()):
    #     print(name,child)
        
    #     if name.find('BatchNorm') != -1:
    #         # print(name)
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         # print(name)
    #         for param in child.parameters():
    #             param.requires_grad = False 
    def freeze_norm_stats(net):
        for m in net.modules():
            m.eval()
            # if isinstance(m, nn.BatchNorm2d):
            #     # print(m)
            #     m.eval()
            # else:
            #     print(m)
            #     m.train()
    model_spin.train()
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            # module.track_running_stats = False
            # module.affine=False
            # print(module)
            # for param in module.parameters():
            #     param.requires_grad = False
   
    # for name, param in model_spin.named_parameters():
    #     print(name)
    #     if name[0:3] in ['fc1','fc2','dec']:
    #         print(name, param.size())
    #         param.requires_grad=True
    #     else:
    #         param.requires_grad=False
    #     # param.requires_grad=False

    def freeze_bn(module):
            '''Freeze BatchNorm layers.'''
            print('module',module)
            # for layer in module:
            #     print('layer',layer)
            #     if isinstance(layer, nn.BatchNorm2d):
            #         layer.eval() 
        
    model_spin.apply(set_bn_eval)
    # model_spin.apply(freeze_norm_stats)
    # print(model_spin)
    # spin_optimizer = model_dict['optimizer_spin']
    spin_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_spin.parameters()), lr=args.lr_spin, weight_decay=0)
    
    if 'nerf' in trainset:
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, amsgrad=True)
        annot_dir_nerf='/media/ExtHDD/Mohsen_data/NerfPose/render_output/'+args.runname+'/poses.npy'
        image_dir_nerf='/media/ExtHDD/Mohsen_data/NerfPose/render_output/'+args.runname+'/image/'
        annot_dir_mpii='/media/ExtHDD/Mohsen_data/mpii_human_pose/mpii_cliffGT.npz'
        image_dir_mpii='/media/ExtHDD/Mohsen_data/mpii_human_pose/'
        # annot_pose=np.load(annot_dir)['pose']
        dataset_mpii=mpii_dataset(annot_dir_mpii,image_dir_mpii)
        data_loader_mpii=DataLoader(dataset_mpii,batch_size=128,num_workers=16)
        dataset_nerf=pose_dataset(annot_dir_nerf,image_dir_nerf)
        data_loader_nerf=DataLoader(dataset_nerf,batch_size=128,num_workers=16)
    
    
    
    criterion_regr = nn.MSELoss().to(device)
    # model_spin=model_spin.float()
    for epoch in range(10):
        print('#epoch:',epoch,'Training on NeRF')
        for idx,sample in enumerate(data_loader_nerf):
            # if idx<len(data_loader_nerf)//5*kk or idx>len(data_loader_nerf)//5*(kk+1):
            #     continue

            image,pose_gt=sample['image'].to('cuda'),sample['pose'].to('cuda')

            # # image,pose_gt=torch.from_numpy(image).float().to('cuda'),torch.from_numpy(pose_gt).float().to('cuda')



            # with torch.no_grad():
            #     model_spin.eval()
            #     pred_rotmat, _, _ = model_spin(image)
            #     pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            #     pose=pose-pose[:,:1]
            #     pose_gt_=pose_gt-pose_gt[:,:1]
            #     frame=100
            #     # fig = plt.figure()
            #     # ax = fig.add_subplot(1,2,1)
            #     # ax.imshow(image[frame].permute(1,2,0).cpu().detach().numpy())
            #     # ax = fig.add_subplot(1,2, 2,projection='3d')
            #     # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            #     # for limb in limbs24:
            #     #     pose_numpy=pose.cpu().detach().numpy()
            #     #     ax.plot(pose_numpy[frame,limb,0],pose_numpy[frame,limb,1],pose_numpy[frame,limb,2],'r')
            #     # for limb in limbs24:
            #     #     pose_gt_numpy=pose_gt_.cpu().detach().numpy()
            #     #     ax.plot(pose_gt_numpy[frame,limb,0],pose_gt_numpy[frame,limb,1],pose_gt_numpy[frame,limb,2],'b')
            #     # ax.set_box_aspect((np.ptp(pose_numpy[frame,:,0]), np.ptp(pose_numpy[frame,:,1]), np.ptp(pose_numpy[frame,:,2]))) 
            #     # plt.show()                  
                
            #     # 2. PA-MPJPE
            #     pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            #     pose_gt_=pose_gt_[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            #     scale_pred=torch.norm(pose,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            #     scale_gt=torch.norm(pose_gt_,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            #     pose=pose/scale_pred*scale_gt
            #     # r_error, pck_error, _ = reconstruction_error(pose, pose_gt2,needpck=True, reduction=None)
            #     # error = np.mean(np.sqrt(((pose - pose_gt2) ** 2).sum(axis=-1)).mean(axis=-1))
            #     spin_loss=torch.mean(torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1),dim=-1)
                
            #     rows1=spin_loss<0.1
            #     rows2=spin_loss>0
            #     rows=rows1*rows2
                
            #     print('mpjpe before fine-runing nerf:',torch.mean(spin_loss[rows1]).item())
            

            
            spin_optimizer.zero_grad()

            # pose_gt=torch3d.pose_gt,scale=0.4)[:,:,:3,-1]
            pred_rotmat, _, _ = model_spin(image)
            pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]  
            
            pose=pose-pose[:,:1]
            pose_gt_=pose_gt-pose_gt[:,:1]
            pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            pose_gt_=pose_gt_[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            scale_pred=torch.norm(pose,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            scale_gt=torch.norm(pose_gt_,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            pose=pose/scale_pred*scale_gt
            # pose=pose-pose[:,:1]
            # pose_gt_=pose_gt[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]    
            # spin_loss=torch.mean(torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1))  
            # spin_loss=F.mse_loss(pose,pose_gt_).double()
            # if i%10==0:
            # spin_loss=criterion_regr(pose[rows].float(),pose_gt_[rows].float())*20
            spin_loss=torch.mean(torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1),dim=-1)*0.1
            rows1=spin_loss<0.0200
            spin_loss=torch.mean(spin_loss[rows1])
            if idx%1==0:
                print('epoch:',epoch,'iteration:',idx,'NeRF spin_loss:',spin_loss.item()) 
            
            spin_loss.backward()
            spin_optimizer.step()
            
        print('#epoch:',epoch,'Training on MPII')
        for idx,sample in enumerate(data_loader_mpii):

            image,pose_gt=sample['image'].to('cuda'),sample['pose'].to('cuda')
            # image,pose_gt=torch.from_numpy(image).float().to('cuda'),torch.from_numpy(pose_gt).float().to('cuda')

            # with torch.no_grad():
            #     model_spin.eval()
            #     pred_rotmat, _, _ = model_spin(image)
            #     pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            #     pose_gt_=get_smpl_l2ws_torch(pose_gt,scale=0.4)[:,:,:3,-1]
            #     pose=pose-pose[:,:1]
            #     pose_gt_=pose_gt_-pose_gt_[:,:1]
                
            #     # pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            #     # pred_vertices = pred_output.vertices
            #     # J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            #     # pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            #     # pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            #     # pred_keypoints_3d = pred_keypoints_3d[:,H36M_TO_J14, :]
            #     # pose = pred_keypoints_3d - pred_pelvis
            #     # pose=pose.cpu().numpy() 
            #     # pose=pose[0]
            #     # print(pose.shape)   
            #     # fig = plt.figure()
            #     # ax = fig.add_subplot(1,2, 2,projection='3d')
            #     # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            #     # for limb in limbs24:
            #     #     pose_numpy=pose.cpu().detach().numpy()
            #     #     ax.plot(pose_numpy[0,limb,0],pose_numpy[0,limb,1],pose_numpy[0,limb,2],'r')
            #     # for limb in limbs24:
            #     #     pose_gt_numpy=pose_gt_.cpu().detach().numpy()
            #     #     ax.plot(pose_gt_numpy[0,limb,0],pose_gt_numpy[0,limb,1],pose_gt_numpy[0,limb,2],'b')
            #     # ax.set_box_aspect((np.ptp(pose_numpy[0,:,0]), np.ptp(pose_numpy[0,:,1]), np.ptp(pose_numpy[0,:,2]))) 
            #     # plt.show()  
                
            #     # 2. PA-MPJPE
            #     pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            #     pose_gt_=pose_gt_[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            #     # r_error, pck_error, _ = reconstruction_error(pose, pose_gt2,needpck=True, reduction=None)
            #     # error = np.mean(np.sqrt(((pose - pose_gt2) ** 2).sum(axis=-1)).mean(axis=-1))
            #     spin_loss=torch.mean(torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1),dim=-1)
                
            #     rows1=spin_loss<0.15
            #     rows2=spin_loss>0.05
            #     rows=rows1*rows2
            
            #     print('mpjpe before fine-runing:',torch.mean(spin_loss[rows]).item())

            # torch.set_grad_enabled(True)
            # model_spin.train()           
            spin_optimizer.zero_grad()
            

            pred_rotmat, _, _ = model_spin(image)
            pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]  
            pose_gt=get_smpl_l2ws_torch(pose_gt,scale=0.4)[:,:,:3,-1] 
            pose=pose-pose[:,:1]
            pose_gt_=pose_gt-pose_gt[:,:1]
            pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            pose_gt_=pose_gt_[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            scale_pred=torch.norm(pose,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            scale_gt=torch.norm(pose_gt_,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            pose=pose/scale_pred*scale_gt

            spin_loss=torch.mean(torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1))*0.1
            # if idx%10==0:
            #     print('epoch:',epoch,' spin_loss:',spin_loss.item()) 
    
            spin_loss.backward()
            spin_optimizer.step()
            
            # with torch.no_grad():
                
            #     model_spin.eval()
            #     pred_rotmat, _, _ = model_spin(image)
            #     pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            #     # fig = plt.figure()
            #     # ax = fig.add_subplot(1,2,1)
            #     # ax.imshow(image[0].permute(1,2,0).cpu().detach().numpy())
            #     # ax = fig.add_subplot(1,2,2,projection='3d')
            #     # # plt.show()
            #     # limbs24=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
            #     # for limb in limbs24:
            #     #     ax.plot(pose[0,limb,0].cpu().detach().numpy(),pose[0,limb,1].cpu().detach().numpy(),pose[0,limb,2].cpu().detach().numpy(),'r')
            #     #     ax.set_box_aspect((np.ptp(pose[0,:,0].cpu().detach().numpy()), np.ptp(pose[0,:,1].cpu().detach().numpy()), np.ptp(pose[0,:,2].cpu().detach().numpy())))
            #     # for limb in limbs24:
            #     #     ax.plot(pose_gt[0,limb,0].cpu().detach().numpy(),pose_gt[0,limb,1].cpu().detach().numpy(),pose_gt[0,limb,2].cpu().detach().numpy(),'b')
            #     #     # ax.set_box_aspect((np.ptp(pose_gt[:,0]), np.ptp(pose_gt[:,1]), np.ptp(pose_gt[:,2])))
            #     # plt.show()
            # #     model_spin.eval()
            # #     pred_rotmat, pred_betas, pred_cam = model_spin(image)
            # #     pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
            #     pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            #     pose_gt_=pose_gt[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]    
            #     spin_loss=torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1)      
            #     print('mpjpe after finetuning:',torch.mean(spin_loss))
            #     print('###################')        
            if idx%10==0:
                print('epoch:',epoch,'iteration:',idx,'MPII spin_loss:',spin_loss.item()) 
      

        if  '3dpw' in evalset:
            Evaluate=evaluate(model=model_spin.eval())
            Evaluate.inference()
        elif 'ski' in evalset:
            evaluate_ski(model=model_spin.eval())

        elif '3dhp' in evalset:
            evaluate_3dhp(model=model_spin.eval())

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_spin.state_dict(),
            'optimizer_state_dict': spin_optimizer.state_dict(),
            },'models/checkpoint_normal'+str(epoch)+'.pth')
    return model_spin

    # pose=pose[0]
    # pose_gt2=pose_gt2[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2, 1)
    # ax.imshow(image2)
    # ax = fig.add_subplot(1,2,2,projection='3d')
    # limbs14=[[0,2],[2,4],[1,3],[3,5],[0,6],[1,6],[6,7],[8,10],[10,12],[9,11],[11,13],[8,6],[9,6]]
    # for limb in limbs14:
    #     ax.plot(pose[limb,0],pose[limb,1],pose[limb,2],'r')
    # ax.set_box_aspect((np.ptp(pose[:,0]), np.ptp(pose[:,1]), np.ptp(pose[:,2]))) 

    # for limb in limbs14:
    #     ax.plot(pose_gt2[limb,0],pose_gt2[limb,1],pose_gt2[limb,2],'b')
    # ax.set_box_aspect((np.ptp(pose[:,0]), np.ptp(pose[:,1]), np.ptp(pose[:,2]))) 

    # plt.show()




def train_spin_ski(model,evalset=['3dpw',],trainset=['nerf',]):
    args=config_parser().parse_args()
    # load data
    data_dict = data_preparation(args)
    model_dict=model_preparation(args)

    model_spin= model.to('cuda')
    
    model_spin.train()
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
        
    model_spin.apply(set_bn_eval)

    spin_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_spin.parameters()), lr=args.lr_spin, weight_decay=0)
    
    if 'nerf' in trainset:
        annot_dir_nerf='/media/ExtHDD/Mohsen_data/NerfPose/render_output/'+args.runname+'/poses.npy'
        image_dir_nerf='/media/ExtHDD/Mohsen_data/NerfPose/render_output/'+args.runname+'/image/'

        dataset_nerf=pose_dataset(annot_dir_nerf,image_dir_nerf)
        data_loader_nerf=DataLoader(dataset_nerf,batch_size=32,num_workers=16)
    
    
    
    criterion_regr = nn.MSELoss().to(device)
    for epoch in range(1):
        for idx,sample in enumerate(data_loader_nerf):

            image,pose_gt=sample['image'].to('cuda'),sample['pose'].to('cuda')

            
            spin_optimizer.zero_grad()

            # pose_gt=torch3d.pose_gt,scale=0.4)[:,:,:3,-1]
            pred_rotmat, _, _ = model_spin(image)
            pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]  
            
            pose=pose-pose[:,:1]
            pose_gt_=pose_gt-pose_gt[:,:1]
            pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            pose_gt_=pose_gt_[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            scale_pred=torch.norm(pose,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            scale_gt=torch.norm(pose_gt_,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            pose=pose/scale_pred*scale_gt
 
            spin_loss=torch.mean(torch.norm(pose - pose_gt_, dim=len(pose_gt_.shape) - 1),dim=-1)
            rows1=spin_loss<0.1
            spin_loss=torch.mean(spin_loss)
            if idx%10==0:
                print('epoch:',epoch,'iteration:',idx,'NeRF spin_loss:',spin_loss.item()) 
            
            spin_loss.backward()
            spin_optimizer.step()

      

        if  epoch>0 and '3dpw' in evalset:
            Evaluate=evaluate(model=model_spin.eval())
            Evaluate.inference()
        if 'ski' in evalset:
            evaluate_ski(model=model_spin.eval())

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': spin_optimizer.state_dict(),
            },'models/checkpoint'+str(epoch)+'.pth')
    return model_spin


def train_gan(args, data_dict,):

    print('###')
    device = torch.device("cuda")

    # bar = Bar('Train pose gan', max=len(data_dict['poses3d_3dpw']))


    for i, inputs_3d in enumerate(data_dict['poses3d_3dpw']):
        rendered_dir = os.path.join(args.outputdir, args.runname, 'image')
        count=len(glob.glob(rendered_dir+'/*'))

        print('???')
    
        outputs_axis_angle = inputs_3d['pose'].to(device)
        outputs_axis_angle=outputs_axis_angle.reshape(-1,24,3)
        print(outputs_axis_angle.shape)
       
        exts=torch.rand(outputs_axis_angle.shape[0],4,4).to(device)-0.5
       
        exts[:,:4,:4]=torch.tensor(
            [[-5.29919172e-01, -5.56525674e-09,  8.48048140e-01, -1.34771157e-07],
            [ 1.47262004e-01 ,  9.84807813e-01 ,  9.20194958e-02, 1.26640154e-08],
            [-8.35164413e-01,  1.73648166e-01, -5.21868549e-01,  4.28571429e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
            ).float()
        c2ws=nerf_extrinsic_to_c2w_torch(exts)
        outputs_axis_angle[:,:1]=(torch.rand(outputs_axis_angle.shape[0],3).to(device)-0.5)*3.14159*2
        print(outputs_axis_angle[:,:1].max(),outputs_axis_angle[:,:1].min())
        outputs_3d_1=get_smpl_l2ws_torch(outputs_axis_angle, scale=0.4)
        outputs_3d_1=outputs_3d_1[:,:,:3,-1].float()
        outputs_2d,outputs_3d = project_to_2d(outputs_3d_1, exts,H=512,W=512,focals=[1000, 1000])
      
                
        pose_gt=outputs_3d
        # limbs=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]

        # pose_gt_p=pose_gt[0].cpu().detach().numpy()
        # # kk=kk[0]
        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection='3d')
        # limbs=[[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[0,3],[3,6],[6,9],[9,14],[14,17],[17,19],[19,21],[21,23],[9,13],[13,16],[16,18],[18,20],[20,22],[9,12],[12,15]]
        # for limb in limbs:
        #     ax.plot(pose_gt_p[limb,0],pose_gt_p[limb,1],pose_gt_p[limb,2])
        # ax.set_box_aspect((np.ptp(pose_gt_p[:,0]), np.ptp(pose_gt_p[:,1]), np.ptp(pose_gt_p[:,2]))) 
        # pose_gt=[] 
        # plt.show()
        # ax = fig.add_subplot(1,3, 2)
        # ax.set_aspect('equal', 'box')
        # target_2d_plot=target_2d[kk].cpu().detach().numpy()
        # ax.scatter(target_2d_plot[:,0],target_2d_plot[:,1])
        
        # for limb in limbs:
        #     ax.plot(target_2d_plot[limb,0],target_2d_plot[limb,1])
        
        # ax = fig.add_subplot(1,3, 3)
        # target_2d_plot=outputs_2d_rt[kk].cpu().detach().numpy()
        # ax.scatter(target_2d_plot[:,0],target_2d_plot[:,1])
        
        # for limb in limbs:
        #     ax.plot(target_2d_plot[limb,0],target_2d_plot[limb,1])
        # # ax.set_box_aspect((np.ptp(target_2d_plot[:,0]), np.ptp(target_2d_plot[:,1])))
        # ax.set_aspect('equal', 'box')
        # plt.show()
        # outputs_axis_angle[200:201,:1]=0.
        run_render(outputs_axis_angle.cpu().detach().numpy(),c2ws.cpu().detach().numpy(),count)
        print('XXX')
        basedir = os.path.join(args.outputdir, args.runname)
        os.makedirs(basedir, exist_ok=True)
        np.save(os.path.join(basedir, 'poses'+str(count)+'.npy'), pose_gt.cpu().detach().numpy(), allow_pickle=True)
        np.save(os.path.join(basedir, 'poses_axis_angles'+str(count)+'.npy'), outputs_axis_angle.cpu().detach().numpy(), allow_pickle=True)
  

   
    return

from torch.utils.data import DataLoader
def data_preparation(args):
    # data_3d_AMASS=np.load('/media/ExtHDD/Mohsen_data/AMASS/processed_AMASS.npz')
    # data_3d_AMASS=torch.from_numpy(data_3d_AMASS['pose3d'][0:len(data_3d_AMASS['pose3d']):10,]).float()
    # data_3d_AMASS_loader = DataLoader(data_3d_AMASS,batch_size=args.batch_size,num_workers=0,
    #                                  shuffle=True, pin_memory=True,drop_last=True)   #,generator=torch.Generator(device='cuda')
    # target_2d=np.load('data/3DPW/3DPW_Validation_2d.npz')
    # target_2d=np.repeat(target_2d['pose2d'],repeats=200,axis=0)
    # target_2d=torch.from_numpy(target_2d)
    # target_2d_loader = DataLoader(target_2d,batch_size=args.batch_size,num_workers=0,
    #                                  shuffle=True, pin_memory=True,drop_last=True)  #,generator=torch.Generator(device='cuda')
    
    pw3d_dataset = PW3D('3dpw')
    pw3d_dataloader = DataLoader(pw3d_dataset, args.batch_size,num_workers=0, shuffle=True)
    
    # print(target_2d.shape)
    # print(data_3d_AMASS.shape)
    return{
        'poses3d_3dpw': pw3d_dataloader,
    } 
Skeleton = namedtuple("Skeleton", ["joint_names", "joint_trees", "root_id", "nonroot_id", "cutoffs", "end_effectors"])

smpl_rest_pose = np.array([[ 0.00000000e+00,  2.30003661e-09, -9.86228770e-08],
                           [ 1.63832515e-01, -2.17391014e-01, -2.89178602e-02],
                           [-1.57855421e-01, -2.14761734e-01, -2.09642015e-02],
                           [-7.04505108e-03,  2.50450850e-01, -4.11837511e-02],
                           [ 2.42021069e-01, -1.08830070e+00, -3.14962119e-02],
                           [-2.47206554e-01, -1.10715497e+00, -3.06970738e-02],
                           [ 3.95125849e-03,  5.94849110e-01, -4.03754264e-02],
                           [ 2.12680623e-01, -1.99382353e+00, -1.29327580e-01],
                           [-2.10857525e-01, -2.01218796e+00, -1.23002514e-01],
                           [ 9.39484313e-03,  7.19204426e-01,  2.06931755e-02],
                           [ 2.63385147e-01, -2.12222481e+00,  1.46775618e-01],
                           [-2.51970559e-01, -2.12153077e+00,  1.60450473e-01],
                           [ 3.83779174e-03,  1.22592449e+00, -9.78838727e-02],
                           [ 1.91201791e-01,  1.00385976e+00, -6.21964522e-02],
                           [-1.77145526e-01,  9.96228695e-01, -7.55542740e-02],
                           [ 1.68482102e-02,  1.38698268e+00,  2.44048554e-02],
                           [ 4.01985168e-01,  1.07928419e+00, -7.47655183e-02],
                           [-3.98825467e-01,  1.07523870e+00, -9.96334553e-02],
                           [ 1.00236952e+00,  1.05217218e+00, -1.35129794e-01],
                           [-9.86728609e-01,  1.04515052e+00, -1.40235111e-01],
                           [ 1.56646240e+00,  1.06961894e+00, -1.37338534e-01],
                           [-1.56946480e+00,  1.05935931e+00, -1.53905824e-01],
                           [ 1.75282109e+00,  1.04682994e+00, -1.68231070e-01],
                           [-1.75758195e+00,  1.04255080e+00, -1.77773550e-01]], dtype=np.float32)

SMPLSkeleton = Skeleton(
    joint_names=[
        # 0-3
        'pelvis', 'left_hip', 'right_hip', 'spine1',
        # 4-7
        'left_knee', 'right_knee', 'spine2', 'left_ankle',
        # 8-11
        'right_ankle', 'spine3', 'left_foot', 'right_foot',
        # 12-15
        'neck', 'left_collar', 'right_collar', 'head',
        # 16-19
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        # 20-23,
        'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ],
    joint_trees=np.array(
                [0, 0, 0, 0,
                 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 9, 9, 12,
                 13, 14, 16, 17,
                 18, 19, 20, 21]),
    root_id=0,
    nonroot_id=[i for i in range(24) if i != 0],
    cutoffs={'hip': 200, 'spine': 300, 'knee': 70, 'ankle': 70, 'foot': 40, 'collar': 100,
            'neck': 100, 'head': 120, 'shoulder': 70, 'elbow': 70, 'wrist': 60, 'hand': 60},
    end_effectors=[10, 11, 15, 22, 23],
)

def get_smpl_l2ws(pose, rest_pose=None, scale=1., skel_type=SMPLSkeleton, coord="xxx",axis_to_matrix=True):
    # TODO: take root as well

    def mat_to_homo(mat):
        last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.concatenate([mat, last_row], axis=0)

    joint_trees = skel_type.joint_trees
    if rest_pose is None:
        # original bone parameters is in (x,-z,y), while rest_pose is in (x, y, z)
        rest_pose = smpl_rest_pose


    # apply scale
    rest_kp = rest_pose * scale
    if axis_to_matrix:
        mrots = [Rotation.from_rotvec(p).as_matrix()  for p in pose]
    else:
        mrots=pose

    mrots = np.array(mrots)

    l2ws = []
    # TODO: assume root id = 0
    # local-to-world transformation
    l2ws.append(mat_to_homo(np.concatenate([mrots[0], rest_kp[0, :, None]], axis=-1)))
    mrots = mrots[1:]
    for i in range(rest_kp.shape[0] - 1):
        idx = i + 1
        # rotation relative to parent
        joint_rot = mrots[idx-1]
        joint_j = rest_kp[idx][:, None]

        parent = joint_trees[idx]
        parent_j = rest_kp[parent][:, None]

        # transfer from local to parent coord
        joint_rel_transform = mat_to_homo(
            np.concatenate([joint_rot, joint_j - parent_j], axis=-1)
        )

        # calculate kinematic chain by applying parent transform (to global)
        l2ws.append(l2ws[parent] @ joint_rel_transform)

    l2ws = np.array(l2ws)

    return l2ws

def train():
    args=config_parser().parse_args()

    fake_3d_sample = Sample_from_Pool()
    fake_2d_sample = Sample_from_Pool()

    args.checkpoint = path.join(datetime.datetime.now().isoformat())
    os.makedirs(args.checkpoint, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(args.checkpoint))

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), args)
    logger.set_names(['epoch'])
    summary = Summary(args.checkpoint)
    writer = summary.create_summary()

    # loss function
    criterion = nn.MSELoss(reduction='mean').to(device)

    # load data
    data_dict = data_preparation(args)
    model_dict=model_preparation(args)
    model_pose=model_dict['model_spin']
    for epoch in range(50):
        train_gan(args, model_dict, data_dict, criterion, fake_3d_sample,fake_2d_sample, summary, writer)
        # if epoch>1:
        #     train_spin(model_pose)
        # if epoch>2:
        #     train_posenet(model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer, criterion, device)
        #     h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
        #                                                             summary, writer, tag='_fake')

        # Update learning rates
        ########################
        model_dict['scheduler_G'].step()
        model_dict['scheduler_d3d'].step()
        print('\nEpoch: %d' % (summary.epoch))

        # Update log file
        logger.append([summary.epoch])

        summary.summary_epoch_update()

    writer.close()
    logger.close()


def run_render(bones,c2ws,count):
    args = config_parser().parse_args()
    # rand_num=np.random.rand()
    # if rand_num<0.5:
    #     args.nerf_args='configs/surreal/surreal.txt' 
    #     args.ckptpath='logs/surreal_model/surreal.tar'
    # elif rand_num>=0.5 and rand_num<0.65:
    #     args.nerf_args='configs/h36m/h36m_prot2.txt' 
    #     args.ckptpath='logs/h36m/s9_200k.tar'
    # elif rand_num>=0.65 and rand_num<0.8:
    #     args.nerf_args='configs/h36m/h36m_prot2.txt' 
    #     args.ckptpath='logs/h36m/s11_200k.tar'  
    # elif rand_num>=0.8 and rand_num<0.9:
    #     args.nerf_args='configs/mixamo/mixamo.txt' 
    #     args.ckptpath='logs/maxima/james_ft_tv.tar' 
    # elif rand_num>=0.9 and rand_num<1:
    #     args.nerf_args='configs/mixamo/mixamo.txt' 
    #     args.ckptpath='logs/maxima/archer_ft_tv.tar' 

    

    # parse nerf model args
    nerf_args = txt_to_argstring(args.nerf_args)
    nerf_args, unknown_args = nerf_config_parser().parse_known_args(nerf_args)
    # print(f'UNKNOWN ARGS: {unknown_args}')

    # load nerf model
    render_kwargs, poseopt_layer = load_nerf(args, nerf_args)

    # prepare the required data
    render_data = load_render_data(args, bones,c2ws,nerf_args, poseopt_layer, nerf_args.opt_framecode)
    tensor_data = to_tensors(render_data)

    basedir = os.path.join(args.outputdir, args.runname)
    os.makedirs(basedir, exist_ok=True)

    rgbs, _, accs, _, bboxes = render_path(render_kwargs=render_kwargs,
                                      chunk=nerf_args.chunk,
                                      ext_scale=nerf_args.ext_scale,
                                      ret_acc=True,
                                      white_bkgd=args.white_bkgd,
                                      **tensor_data)

    if args.no_save:
        return

    rgbs = (rgbs * 255).astype(np.uint8)
    accs = (accs * 255).astype(np.uint8)
    skeletons = draw_skeletons_3d(rgbs, render_data['kp'],
                                  render_data['render_poses'],
                                  *render_data['hwf'])
    
    os.makedirs(os.path.join(basedir, 'image'), exist_ok=True)

    for i, rgb in enumerate(rgbs):
            ii=count+i
            imageio.imwrite(os.path.join(basedir, 'image', f'{ii:05d}.png'), rgb)
            # image=mpimg.imread(os.path.join(basedir, 'image', f'{i:05d}.png'))
            # plt.imshow(image)
            # plt.show()   

    # np.save(os.path.join(basedir, 'poses.npy'), bones, allow_pickle=True)
    # imageio.mimwrite(os.path.join(basedir, "render_rgb.mp4"), rgbs, fps=args.fps)


from skimage.transform import resize  
# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_RES = 224
args = config_parser().parse_args()
IMG_DIR= 'render_output/'+args.runname+'/image'
normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_MEAN)

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0, bgmask=None):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)
    if bgmask is not None:
        new_bgmask = np.zeros([br[1] - ul[1], br[0] - ul[0]])

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]
    if bgmask is not None:
        new_bgmask[new_y[0]:new_y[1], new_x[0]:new_x[1]] = bgmask[old_y[0]:old_y[1], 
                                                                  old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        # new_img = scipy.misc.imrotate(new_img, rot)
        new_img = scipy.ndimage.interpolation.rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
        if bgmask is not None:
            new_bgmask = scipy.ndimage.interpolation.rotate(new_bgmask, rot)
            new_bgmask = new_bgmask[pad:-pad, pad:-pad]

    # new_img = scipy.misc.imresize(new_img, res)
    
    new_img = resize(new_img, res)
    if bgmask is not None:
        new_bgmask = resize(new_bgmask, res)
        return new_img, new_bgmask
    else:
        return new_img

def process_sample(image):
    img = rgb_processing(image)
    img = torch.from_numpy(img).cpu()
    img = normalize_img(img)
    return img

def read_image(imgname):
    # imgname = os.path.join(IMG_DIR, imgname)
    img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
    return img

def rgb_processing(rgb_img):
    # rgb_img = crop(rgb_img.copy(), center, scale, [IMG_RES, IMG_RES], rot=rot)
    # if is_train and flip:
    #         rgb_img = flip_img(rgb_img)
    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    return rgb_img

import json
def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""

    center = openpose_file.mean(axis=0)
    bbox_size = (openpose_file.max(axis=0) - openpose_file.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale


def process_image(img_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment

    center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img



import glob

if __name__ == '__main__':
    args=config_parser().parse_args()
    data_dict = data_preparation(args)
    train_gan(args,data_dict)