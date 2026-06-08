from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / 'Dataset'
CHECKPOINT_ROOT = PROJECT_ROOT / 'checkpoint'

PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = str(CHECKPOINT_ROOT / 'diff') + '/'
OUT_PATH = str(CHECKPOINT_ROOT / 'pd_score' / 'logs') + '/'

# KINECT
PREPROCESSED_DATA_ROOT_PATH = str(DATASET_ROOT)

# PD
PD_PATH_POSES = str(DATASET_ROOT / 'motion3d' / 'MB3D_f81s9')
PD_PATH_LABELS = str(DATASET_ROOT / 'motion3d' / 'MB3D_f81s9')

CHECKPOINT_ROOT_PATH = str(CHECKPOINT_ROOT / 'pd_score') + '/'
