## how to use the code:
# nohup python run_gan.py --nerf_args configs/surreal/surreal.txt --ckptpath logs/surreal_model/surreal.tar  --dataset surreal --entry hard  --runname render_3dpw_testset --white_bkgd  --render_res 512 512 > run_train_rendered_3dpw_testset.out 2>&1 &
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
import glob

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
from pytorch_msssim import SSIM
import torch.nn as nn
import time
from progress.bar import Bar
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
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

    parser.add_argument('--batch_size', type=int, default=1024,
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
    parser.add_argument('--outputdir', type=str, default='render_output/',
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

    
    bones=bones
    cam_idxs=np.arange(bones.shape[0])
    focals = focals[cam_idxs]
    c2ws = c2ws[cam_idxs]
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, 1.0) for bone in bones])
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



class PoseGenerator(nn.Module):
    def __init__(self, args, input_size=24 * 3):
        super(PoseGenerator, self).__init__()
        self.BAprocess = BAGenerator(input_size=24 * 3,noise_channle=32)
        self.RTprocess = RTGenerator(input_size=24 * 3) #target

    def forward(self, inputs_3d):
        '''
        input: 3D pose
        :param inputs_3d: nx16x3, with hip root
        :return: nx16x3
        '''
        
        pose_ba = self.BAprocess(inputs_3d)  # diff may be used for div loss
        R,T,pose_rt = self.RTprocess(inputs_3d)  # rt=(r,t) used for debug

        return {'pose_ba': pose_ba,
                'ba_diff': None,
                'pose_bl': None,
                'blr': None,
                'pose_rt': pose_rt,
                'R': R,
                'T': T}


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

        noise = torch.randn(inputs_3d.shape[0], self.noise_channle, device=inputs_3d.device) #*2-1

        # noise=self.Sampler(inputs_3d.shape[0],device=inputs_3d.device)
        # noise = noise / noise.norm(dim=1, keepdim=True)

        y = self.w1(noise) 
  
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
       

    def forward(self, input_3d):
        x1= self.layer_left_leg(input_3d[:,[4,7,10]].reshape(-1,3*3))
        x2= self.layer_right_leg(input_3d[:,[5,8,11]].reshape(-1,3*3))
        x3= self.layer_left_arm(input_3d[:,[9,13,16,18,20,22]].reshape(-1,6*3))
        x4= self.layer_right_arm(input_3d[:,[9,14,17,19,21,23]].reshape(-1,6*3))
        x5= self.layer_torso(input_3d[:,[0,1,2,3,6,9,13,14,16,17]].reshape(-1,10*3))
        x6= self.layer_head(input_3d[:,[9,12,15]].reshape(-1,3*3))
        x7= self.layer_full_body(input_3d.reshape(-1,24*3))
        
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
    checkpoint = torch.load('models/checkpoint_normal9.pth') #'data/data/model_checkpoint.pt' 'models/checkpoint4.pth'
    model_spin.load_state_dict(checkpoint['model_state_dict'], strict=False) #'model' 'model_state_dict'

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

    adv_3d_real_loss = criterion(real_3d, fake_label_3d)
    adv_3d_fake_loss = criterion(fake_3d, real_label_3d)
    # Total discriminators losses
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
        
        print("== Final Results ==")
        print('MPJPE:', mpjpe*1000)
        print('PAMPJPE:', pampjpe*1000)
        print('Mesh Error:', uposed_mesh_error, posed_mesh_error)



    def test(self, databatch):
        
        gt_pose = databatch['pose']
        gt_betas = databatch['betas']
        gender = databatch['gender']            
        
        with torch.no_grad():
            # forward
            self.model.eval()
            self.model2.eval()
            images = databatch['image']
            pred_rotmat, pred_betas, pred_cam = self.model(images)
            pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']


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
        return len(self.annot)
    
    def __getitem__(self,idx):
        idx_=idx    
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

def train_spin(model,evalset=['3dpw',],trainset=['nerf',]):
    args=config_parser().parse_args()
   
    model_spin= model.to('cuda')
    
   
    def freeze_norm_stats(net):
        for m in net.modules():
            m.eval()
        
    model_spin.train()
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


    def freeze_bn(module):
            '''Freeze BatchNorm layers.'''
            print('module',module)
 
    model_spin.apply(set_bn_eval)

    spin_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_spin.parameters()), lr=args.lr_spin, weight_decay=0)
    
    if 'nerf' in trainset:
        annot_dir_nerf='render_output/'+args.runname+'/poses.npy'
        image_dir_nerf='render_output/'+args.runname+'/image/'
        annot_dir_mpii='data/mpii_human_pose/mpii_cliffGT.npz'
        image_dir_mpii='data/mpii_human_pose/'
        dataset_mpii=mpii_dataset(annot_dir_mpii,image_dir_mpii)
        data_loader_mpii=DataLoader(dataset_mpii,batch_size=128,num_workers=16)
        dataset_nerf=pose_dataset(annot_dir_nerf,image_dir_nerf)
        data_loader_nerf=DataLoader(dataset_nerf,batch_size=128,num_workers=16)
    
    
    
    criterion_regr = nn.MSELoss().to(device)
    for epoch in range(10):
        print('#epoch:',epoch,'Training on NeRF')
        for idx,sample in enumerate(data_loader_nerf):


            image,pose_gt=sample['image'].to('cuda'),sample['pose'].to('cuda')
            
            spin_optimizer.zero_grad()

            pred_rotmat, _, _ = model_spin(image)
            pose=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]  
            
            pose=pose-pose[:,:1]
            pose_gt_=pose_gt-pose_gt[:,:1]
            pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            pose_gt_=pose_gt_[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            scale_pred=torch.norm(pose,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            scale_gt=torch.norm(pose_gt_,dim=(-1,-2)).unsqueeze(1).unsqueeze(1)
            pose=pose/scale_pred*scale_gt
        
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
       
            spin_loss.backward()
            spin_optimizer.step()

            if idx%10==0:
                print('epoch:',epoch,'iteration:',idx,'MPII spin_loss:',spin_loss.item()) 
      

        if  '3dpw' in evalset:
            Evaluate=evaluate(model=model_spin.eval())
            Evaluate.inference()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_spin.state_dict(),
            'optimizer_state_dict': spin_optimizer.state_dict(),
            },'models/checkpoint_normal'+str(epoch)+'.pth')
    return model_spin



def train_gan(args, model_dict, data_dict, criterion, fake_3d_sample,fake_2d_sample, summary, writer):
    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # extract necessary module for training.
    model_G = model_dict['model_G']
    model_d3d = model_dict['model_d3d']
    model_d2d = model_dict['model_d2d']
    model_spin= model_dict['model_spin']       

    g_optimizer = model_dict['optimizer_G']
    d3d_optimizer = model_dict['optimizer_d3d']
    d2d_optimizer = model_dict['optimizer_d2d']
    spin_optimizer = model_dict['optimizer_spin']     

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_G.train()
    model_d3d.train()
    model_d2d.train()
    model_spin.train()    
    end = time.time()

    # prepare buffer list for update
    tmp_3d_pose_buffer_list = []
    tmp_2d_pose_buffer_list = []
    tmp_camparam_buffer_list = []

    bar = Bar('Train pose gan', max=len(data_dict['target_2d']))

    SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'
    model = hmr(SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load('data/data/model_checkpoint.pt')
    model.load_state_dict(checkpoint['model'], strict=False)

    # count=summary.epoch*21

    for i, (inputs_3d,target_2d) in enumerate(zip(data_dict['poses3d_AMASS'],data_dict['target_2d'])):
        rendered_dir = os.path.join(args.outputdir, args.runname, 'image')
        count=len(glob.glob(rendered_dir+'/*'))//args.rpi
        lr_now = g_optimizer.param_groups[0]['lr']

        ##################################################
        #######      Train Generator     #################
        ##################################################
        set_grad([model_d3d], False)
        set_grad([model_d2d], False)
        set_grad([model_G], True)
        set_grad([model_spin], False)
        g_optimizer.zero_grad()

        # Measure data loading time
        data_time.update(time.time() - end)
       
        inputs_3d, target_2d = inputs_3d.to(device), target_2d.to(device)
        target_2d=target_2d.float()

        # Generator
        g_rlt = model_G(inputs_3d)

        # extract the generator result
        outputs_axis_angle = g_rlt['pose_ba']

        outputs_R = g_rlt['R']
        outputs_T = g_rlt['T']
        exts=torch.rand(inputs_3d.shape[0],4,4).to(device)-0.5
      
        exts[:,:4,:4]=torch.tensor(
            [[-5.29919172e-01, -5.56525674e-09,  8.48048140e-01, -1.34771157e-07],
            [ 1.47262004e-01 ,  9.84807813e-01 ,  9.20194958e-02, 1.26640154e-08],
            [-8.35164413e-01,  1.73648166e-01, -5.21868549e-01,  4.28571429e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
            ).float()
        c2ws=nerf_extrinsic_to_c2w_torch(exts)

     
        outputs_3d_1=get_smpl_l2ws_torch(outputs_axis_angle, scale=0.4)
        outputs_3d_1=outputs_3d_1[:,:,:3,-1].float()
        outputs_2d,outputs_3d = project_to_2d(outputs_3d_1, exts,H=512,W=512,focals=[1000, 1000])
 
        # adv loss
        adv_3d_loss = get_adv_loss(model_d3d, inputs_3d, outputs_axis_angle, criterion, summary, writer, writer_name='g3d')


        # # plot a image for visualization
        if i%5==0 and summary.epoch>2:
            kk=torch.randint(1024,(args.rpi,))
            
            # kk=[200,400]
            pose_gt=outputs_3d[kk]
        
            run_render(outputs_axis_angle[kk].cpu().detach().numpy(),c2ws[kk].cpu().detach().numpy(),count)
            
            basedir = os.path.join(args.outputdir, args.runname)
            os.makedirs(basedir, exist_ok=True)
            np.save(os.path.join(basedir, 'poses'+str(count)+'.npy'), pose_gt.cpu().detach().numpy(), allow_pickle=True)
            np.save(os.path.join(basedir, 'poses_axis_angles'+str(count)+'.npy'), outputs_axis_angle.cpu().detach().numpy(), allow_pickle=True)

            files=glob.glob('render_output/'+args.runname+'/image/*')
            
            pose=torch.zeros_like(pose_gt,device='cuda')
            for ii in range(args.rpi):
                jj=ii+count
                # ii=int(file[-8:-4])
                file=os.path.join(basedir, 'image', f'{jj:05d}.png')
                image = read_image(file)
                image = image[100:412,100:412,:]
                image = process_sample(image.copy())
                image = resize(image, (3,224, 224),
                                anti_aliasing=True)
                
                # Cropped image of above dimension
                # (It will not change original image)
                image2=np.transpose(image.astype('float32'),(1,2,0))

                image=torch.from_numpy(image).float()
                image=image.unsqueeze(0)
                
                data_loader=DataLoader(image,batch_size=1)
                for _,sample in enumerate(data_loader):
                    sample=sample.to('cuda')
                    # print(sample.shape)
                    with torch.no_grad():
                        model.eval()
                        pred_rotmat, pred_betas, pred_cam = model(sample)
                        pose[ii]=get_smpl_l2ws_torch(pred_rotmat,axis_to_matrix=False,scale=0.4)[:,:,:3,-1]
              
                        
            # 2. PA-MPJPE
            pose=pose-pose[:,:1]
            pose_gt=pose_gt-pose_gt[:,:1]
            pose=pose[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            pose_gt=pose_gt[:,[1,2,4,5,7,8,12,15,16,17,18,19,20,21]]
            # r_error, pck_error, _ = reconstruction_error(pose, pose_gt2,needpck=True, reduction=None)
            # error = np.mean(np.sqrt(((pose - pose_gt2) ** 2).sum(axis=-1)).mean(axis=-1))
            spin_loss=1-mpjpe(pose,pose_gt)

  
        else:
            spin_loss=0

        results = {'mpjpe':spin_loss,'adv_3d_loss':adv_3d_loss.item()}
        print(results)
        gen_loss =  adv_3d_loss  +\
                    spin_loss*0.1 

    
        # Update generators
        ###################################################
        gen_loss.backward()
        nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1)
        g_optimizer.step()

        ##################################################
        #######      Train Discriminator     #############
        ##################################################
        if summary.train_iter_num % args.df == 0:
            set_grad([model_d3d], True)
            set_grad([model_d2d], True)
            set_grad([model_G], False)

            # d3d training

            train_dis(model_d3d, inputs_3d, outputs_axis_angle, criterion, summary, writer, writer_name='d3d',
                      fake_data_pool=fake_3d_sample, optimizer=d3d_optimizer)
 


        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=len(data_dict['poses3d_AMASS']), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    bar.finish()

    return

            

from torch.utils.data import DataLoader
def data_preparation(args):
    data_3d_AMASS=np.load('/data/AMASS/processed_AMASS.npz')
    data_3d_AMASS=torch.from_numpy(data_3d_AMASS['pose3d'][0:len(data_3d_AMASS['pose3d']):10,]).float()
    data_3d_AMASS_loader = DataLoader(data_3d_AMASS,batch_size=args.batch_size,num_workers=0,
                                     shuffle=True, pin_memory=True,drop_last=True)   #,generator=torch.Generator(device='cuda')
    target_2d=np.load('data/3DPW/3DPW_Validation_2d.npz')
    target_2d=np.repeat(target_2d['pose2d'],repeats=200,axis=0)
    target_2d=torch.from_numpy(target_2d)
    target_2d_loader = DataLoader(target_2d,batch_size=args.batch_size,num_workers=0,
                                     shuffle=True, pin_memory=True,drop_last=True)  #,generator=torch.Generator(device='cuda')

    return{
        'poses3d_AMASS': data_3d_AMASS_loader,
        'target_2d': target_2d_loader
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
            ii=count*args.rpi+i
            imageio.imwrite(os.path.join(basedir, 'image', f'{ii:05d}.png'), rgb)


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
        new_img = scipy.ndimage.interpolation.rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
        if bgmask is not None:
            new_bgmask = scipy.ndimage.interpolation.rotate(new_bgmask, rot)
            new_bgmask = new_bgmask[pad:-pad, pad:-pad]

    
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
    img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
    return img

def rgb_processing(rgb_img):
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



if __name__ == '__main__':
    args=config_parser().parse_args()
    train()
   