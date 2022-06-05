''' Detecting STAS on Public and Private image Dataset
    Command: python demo_stas_obj.py --weight_dir htc_swin_s_7_088_Weights --testdata_dirs Public_Image Private_Image/Image
'''

print("Running ....")

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot

import os
import cv2
import glob
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--weight_dir', required=True, type=str, help="Pretrained weights directory to load. Please put under /Weights folder")
parser.add_argument('--testdata_dirs', required=True, type=str, nargs='+', help="Pretrained datasets directories to load. Please put under /Datasets folder")
args = parser.parse_args()

#### Loading directories path

current_dir_root = os.getcwd()
weight_dir_root = args.weight_dir
testdata_dir_root = args.testdata_dirs
# weight_dir_root = "htc_swin_s_7_088_Weights"
# testdata_dir_root = ["Public_Image", "Private_Image/Image"]


#### Load model and config

model = pickle.load(open(current_dir_root + '/Weights/' + weight_dir_root + '/model.pkl','rb'))
cfg = pickle.load(open(current_dir_root + '/Weights/' + weight_dir_root + '/config.pkl','rb'))
model.cfg = cfg


#### Build the detector "with loading checkpoints"

checkpoint = current_dir_root + '/Weights/' + weight_dir_root + '/best_bbox_mAP_epoch_30.pth'
checkpoint = load_checkpoint(model, checkpoint, map_location='cuda:0')


#### Parse and Save the result

mmcv.mkdir_or_exist(current_dir_root + "/Output")
mmcv.mkdir_or_exist(current_dir_root + "/Output/segm")
mmcv.mkdir_or_exist(current_dir_root + "/Output/inst_segm")

result = {}
for folder in testdata_dir_root:
    for file in sorted(glob.glob(current_dir_root + "/Datasets/" + folder + "/*.jpg")):

        img_filename = file.rsplit('/', 1)[1]
        print("\n", img_filename)

        output = inference_detector(model, file)


        ### BndBox Result

        print("\n>>>>>>>> BndBox")

        bndbox = []
        for j in output[0][0]:

            if float(j[-1]) >= 0.05:

                temp = list(map(int, j[:-1]))
                temp.append(round(float(j[-1]), 5))
                print(temp)
                bndbox.append(temp)

        result.update({img_filename: bndbox})


        ### Segmentation Mask Result

        print("\n>>>>>>>> Segmentation Mask")

        if len(mmcv.concat_list(output[1])) > 0:

            segms = np.stack(mmcv.concat_list(output[1]), axis=0)
            result_mask = np.zeros(segms[0].shape, dtype="uint8")
            n = len(segms)

            # Overlapped segmentation result
            for k, seg in enumerate(segms):
                result_mask[seg > 0] = 255
                cv2.imwrite(current_dir_root + "/Output/segm/" + img_filename.rsplit('.', 1)[0] + ".png", result_mask)
            print(result_mask.shape, " Unique: ", np.unique(result_mask)) # Check if only contains 0, 255


        ### Save result image ( Bndbox with Segmentation )

        model.show_result(file, output, 
                          out_file=current_dir_root + "/Output/inst_segm/" + img_filename.rsplit('.', 1)[0] +".jpg")
        
with open(current_dir_root + "/Output/result_bbox.json", 'w') as fp:
    json.dump(result, fp)



