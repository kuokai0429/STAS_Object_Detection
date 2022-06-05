# STAS_Object_Detection
 肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 I：運用物體偵測作法於找尋STAS <br>

## -------- Object Detection -------- 

### Environment (TWCC Container): 

pytorch-21.06-py3:latest <br>

### Command: 

git clone https://github.com/kuokai0429/STAS_Object_Detection.git <br>
pip3 install pipenv <br>
cd STAS_Object_Detection <br>
pipenv --python 3.8 <br>
pipenv shell <br>
pipenv install --skip-lock <br>
git clone https://github.com/open-mmlab/mmdetection.git <br>
cd mmdetection <br>
python setup.py develop <br>
cd .. <br>
sudo apt-get update <br>
sudo apt-get install ffmpeg libsm6 libxext6  -y <br>
python demo_coco.py <br>
