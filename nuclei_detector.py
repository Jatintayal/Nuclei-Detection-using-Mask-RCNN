import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys
sys.path.append('../')

from config2 import *
from dataset import NucleiDataset
import numpy as np
import model as modellib
from model import log
import utils
import random

# Training dataset
dataset_train = NucleiDataset()
dataset_train.add_nuclei(opt.train_data_root,'train')
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset()
dataset_val.add_nuclei(opt.val_data_root,'val')
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=opt,
                          model_dir=opt.MODEL_DIR)

init_with = opt.init_with  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    if not os.path.exists(opt.COCO_MODEL_PATH):
        utils.download_trained_weights(opt.COCO_MODEL_PATH)
    
    model.load_weights(opt.COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=opt.LEARNING_RATE,
            epochs=40,
            layers='all')

def dilation(mask):
    return binary_dilation(mask, disk(1))
    
class InferenceConfig(Config2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400
    
    inference_config = InferenceConfig()
    
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=opt.MODEL_DIR)

    model_path = model.find_last()[1]
    
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
        
		
for image_id in dataset_val.image_ids:#random.choice(dataset_val.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask, original_size =\
            modellib.load_image_gt2(dataset_val, inference_config, 
                                   image_id, use_mini_mask=False)
        
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    
        
    #resize and resize back
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset_train.class_names,
                                    figsize=(18, 18), plot_dir=plot_dir, im_id=image_id, alt='gt')
    print(original_size.shape)
    print(original_image.shape)
    print(gt_mask.shape)#(256, 256, 22)
        
    #display original masks in one figure wihout resize
    original_mask = imageio.imread(dataset_val.image_info[image_id]['mask_dir'].replace('masks/','images/') + 'mask.png')
    #plt.figure(figsize=(18, 18))
    #plt.imshow(original_mask)#, cmap='gray')
	print(original_mask.shape)#(360, 360)
        
        '''
        visualize val predict
        '''
    results = model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                        dataset_val.class_names, r['scores'],
                                        figsize=(18, 18), plot_dir=plot_dir, im_id=image_id, alt='p')
    print(r['masks'].shape)
		
		
ths = np.linspace(0.5,0.95,10)
image_ids = dataset_val.image_ids
    
APs = []
for i, image_id in enumerate(image_ids):
    #print(i)
        
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    AP = []
    for th in ths:
        AP_th =\
                utils.compute_metric_masks(gt_mask,
                                 r['masks'],
                                 iou_threshold=th)
        AP.append(AP_th)
    APs.append(AP)
        
print("mAP: ", np.mean(APs))