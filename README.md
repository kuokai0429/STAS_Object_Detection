# STAS_Object_Detection

### About:

肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 I：運用物體偵測作法於找尋STAS <br>

### Environment (TWCC Container): 

pytorch-21.06-py3:latest <br>

### Command: 

```
git clone https://github.com/kuokai0429/STAS_Object_Detection.git
pip3 install pipenv
cd STAS_Object_Detection
pipenv --python 3.8
pipenv shell
pipenv install --skip-lock
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python setup.py develop
cd ..
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
python demo_coco.py
```
