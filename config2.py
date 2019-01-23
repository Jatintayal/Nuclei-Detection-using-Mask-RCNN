import sys
sys.path.append('../')
import config
import numpy as np


class Config2(config.Config):

    # Give the configuration a recognizable name
    NAME = "nuclei"
    train_data_root = 'data/stage1_train/' 
    val_data_root = 'data/stage1_train/'
    test_data_root = 'data/stage1_test/'
    
    MODEL_DIR = 'checkpoints'
    COCO_MODEL_PATH = 'mask_rcnn_coco.h5'
    # imagenet, coco, or last
    init_with = "imagenet"  
    
    LEARNING_RATE = 1e-4
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Train on 1 GPU and 8 images per GPU. Batch size is GPUs * images/GPU.
    GPU_COUNT = 4
    IMAGES_PER_GPU = 2
    bs = GPU_COUNT * IMAGES_PER_GPU
    # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
    # typically be equal to the number of samples of your dataset divided by the batch size
    STEPS_PER_EPOCH = 600//bs
    VALIDATION_STEPS = 70//bs

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
	
   # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported



    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels, maybe add a 256?
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320 #300
    
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    TRAIN_ROIS_PER_IMAGE = 512
    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce(increase?) this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 256
    
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400 
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3 # 0.3
    
    
    MEAN_PIXEL = np.array([42.17746161,38.21568456,46.82167803])
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    
    
opt = Config2()



if __name__ == '__main__':
    opt.display()