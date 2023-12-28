# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}


JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

JOINT_REGRESSOR_TRAIN_EXTRA = 'data/spin_data/J_regressor_extra.npy'
SMPL_MODEL_DIR = 'data/spin_data/'
JOINT_REGRESSOR_H36M = 'data/spin_data/J_regressor_h36m.npy'

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_RES = 224

import os.path as osp


SMPL_MODEL_DIR = 'data/smpl'
SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'

PW3D_ROOT = 'data/3DPW/'
MPI_INF_3DHP_ROOT = 'path_to_mpi-inf-3dhp'
H36M_ROOT = 'path_to_human36m'

# folder to save processed files
DATASET_NPZ_PATH = 'data/dataset_extras'
PW3D_ANNOT_DIR = osp.join(DATASET_NPZ_PATH, '3dpw_vid')

JOINT_REGRESSOR_TRAIN_EXTRA = 'data/spin_data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/spin_data/J_regressor_h36m.npy'

# Path to test/train npz files
DATASET_FILES = [ {'h36m': osp.join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'mpi-inf-3dhp': osp.join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
                   '3dpw': osp.join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                  },

                  {'h36m': osp.join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   '3dpw': osp.join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'mpi-inf-3dhp': osp.join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz')
                  }
                ]

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

pw3d_annot_names = [
    'downtown_runForBus_00',
    'downtown_rampAndStairs_00',
    'flat_packBags_00',
    'downtown_runForBus_01',
    'office_phoneCall_00', 
    'downtown_windowShopping_00',
    'downtown_walkUphill_00', 
    'downtown_sitOnStairs_00', 
    'downtown_enterShop_00', 
    'downtown_walking_00', 
    'downtown_stairs_00', 
    'downtown_crossStreets_00', 
    'downtown_car_00', 
    'downtown_downstairs_00', 
    'downtown_bar_00', 
    'downtown_walkBridge_01', 
    'downtown_weeklyMarket_00', 
    'downtown_warmWelcome_00', 
    'downtown_arguing_00', 
    'downtown_upstairs_00', 
    'downtown_bus_00', 
    'flat_guitar_01', 
    'downtown_cafe_00', 
    'outdoors_fencing_01'
    ]

J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]

