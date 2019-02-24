import os, sys, tarfile, zipfile, time, argparse, subprocess
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mrcnn.model import log
from mrcnn import visualize, utils
import mrcnn.model as modellib
import nucleus

path = os.getcwd()
model_tar = "nuclei_datasets.tar.gz"
data_path = os.path.join(path + '/nuclei_datasets')
model_path = os.path.join(path + '/logs/nucleus')
weights_path = os.path.join(model_path + '/mask_rcnn_nucleus.h5')

DEVICE = "/gpu:0"   # If there is no gpu, change to cpu

config = nucleus.NucleusConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
dataset = nucleus.NucleusDataset()

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)

model.load_weights(weights_path, by_name=True)

parser = argparse.ArgumentParser(description='Do you want to train or detect?')
parser.add_argument("command", metavar="<command>", help="'train' or 'detect' or plot mAP with 'mAP-train' or 'mAP-val'")
parser.parse_args()

def unzipper():
    tar_file = tarfile.open(model_tar)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'nuclei_datasets' not in file_name:
            tar_file.extract(file, os.getcwd())

    os.chdir(path + "/nuclei_datasets")
    with zipfile.ZipFile("stage1_test.zip","r") as zip_ref:
        zip_ref.extractall("stage1_test")
    with zipfile.ZipFile("stage1_train.zip","r") as zip_ref:
        zip_ref.extractall("stage1_train")

    os.remove("stage1_test.zip")
    os.remove("stage1_train.zip")
    return

def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        AP = 1 - AP
        APs.append(AP)
    return APs, precisions, recalls

if not os.path.exists(path + "/nuclei_datasets"):
    unzipper()
    print('Unzipping successful.')

os.chdir(path)
if sys.argv[1] == 'train':
    subprocess.Popen("py nucleus.py train --dataset=nuclei_datasets --subset=stage1_train --weights=last --logs=logs", shell=True)

elif sys.argv[1] == 'test':
    subprocess.Popen("py nucleus.py detect --dataset=nuclei_datasets --subset=stage1_test --weights=logs/nucleus/mask_rcnn_nucleus.h5", shell=True)

elif sys.argv[1] == 'mAP-val':
    dataset.load_nucleus(data_path, 'val')
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    print("Loading weights ", weights_path)

    image_ids = np.random.choice(dataset.image_ids, 25)
    APs, precisions, recalls = compute_batch_ap(image_ids)
    print("mAP @ IoU=50: ", APs)

    AP = np.mean(APs)
    visualize.plot_precision_recall(AP, precisions, recalls)
    plt.show()

elif sys.argv[1] == 'mAP-train':
    dataset.load_nucleus(data_path, 'train')
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    print("Loading weights ", weights_path)
    
    num = input('\nSince plotting mAP for all training images will take time, Please provide a number that is viable -- ')
    
    image_ids = np.random.choice(dataset.image_ids, int(num))
    APs, precisions, recalls = compute_batch_ap(image_ids)
    print("mAP @ IoU=50: ", APs)
    
    AP = np.mean(APs)
    visualize.plot_precision_recall(AP, precisions, recalls)
    plt.show()