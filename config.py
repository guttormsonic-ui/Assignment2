import torch
import os

DATA_DIR = 'D:\\ML\\A2'
#Full Set
'''OXFORD_PET_CLASSES = ['__background__', 'Abyssinian', 'American_Bulldog', 
'American_Pit_Bull_Terrier', 'Basset_Hound', 'Beagle', 'Bengal', 'Birman', 
'Bombay', 'Boxer', 'British_Shorthair', 'Chihuahua', 'Egyptian_Mau', 
'English_Cocker_Spaniel', 'English_Setter', 'Exotic_Shorthair', 
'German_Shorthaired_Pointer', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin', 
'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland',
'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard', 
'Samoyed', 'Scottish_Terrier', 'Siberian_Husky', 'Sphynx', 'Staffordshire_Bull_Terrier',
'Siamese', 'Shiba_Inu', 'Wheaten_Terrier']'''

# Oxford-IIIT Pet Dataset - 10 breeds
OXFORD_PET_DIR = os.path.join(DATA_DIR, 'Pet')
OXFORD_PET_CLASSES = ['__background__', 'chihuahua', 'wheaten terrier',
 'english cocker spaniel', 
'shiba inu', 'boxer', 'samoyed', 'pomeranian','Pug', 'british shorthair','Russian_Blue']
NUM_OXFORD_PET_CLASSES = len(OXFORD_PET_CLASSES)

# Penn-Fudan Pedestrian Dataset
PENN_FUDAN_DIR = os.path.join(DATA_DIR, 'PF')
PENN_FUDAN_CLASSES = ['__background__', 'pedestrian']
NUM_PENN_FUDAN_CLASSES = len(PENN_FUDAN_CLASSES)

# Models
FASTER_RCNN_MODEL_NAME = 'fasterrcnn_resnet50_fpn'
YOLOV5N_MODEL_NAME = 'yolov5n'

#SELECT MODELS AND DATASETS
ACTIVE_MODEL = YOLOV5N_MODEL_NAME # Options: FASTER_RCNN_MODEL_NAME, YOLOV5N_MODEL_NAME
ACTIVE_DATASET = 'Oxford-IIIT Pet'     # Options: 'Oxford-IIIT Pet', 'Penn-Fudan Pedestrian'

ACTIVE_DATASET_ROOT = None
ACTIVE_NUM_CLASSES = None
ACTIVE_CLASSES_LIST = None

if ACTIVE_DATASET == 'Oxford-IIIT Pet':
    ACTIVE_DATASET_ROOT = OXFORD_PET_DIR
    ACTIVE_NUM_CLASSES = NUM_OXFORD_PET_CLASSES
    ACTIVE_CLASSES_LIST = OXFORD_PET_CLASSES
elif ACTIVE_DATASET == 'Penn-Fudan Pedestrian':
    ACTIVE_DATASET_ROOT = PENN_FUDAN_DIR
    ACTIVE_NUM_CLASSES = NUM_PENN_FUDAN_CLASSES
    ACTIVE_CLASSES_LIST = PENN_FUDAN_CLASSES
else:
    raise ValueError(f"Unknown ACTIVE_DATASET specified in config.py: {ACTIVE_DATASET}")

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
LOG_DIR = os.path.join(DATA_DIR, 'logs')