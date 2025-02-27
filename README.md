# FasDG


## Environmental Setup

- Our code is based on MMdetection 2.26.0

- Install the required packages according to `https://github.com/open-mmlab/mmdetection/tree/v2.26.0`

## Dataset Preparation

  Download the [Urban scene dataset](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B) and modify the file path in `configs\_urban_scene\dataset_urban_scene` as shown in:

```
data_root_test_t1 = 'xxx/Daytime_Sunny/VOC2007/'
data_root_test_t2 = 'xxx/Daytime-Foggy/VOC2007/'
data_root_test_t3 = 'xxx/Dusk-rainy/VOC2007/'
data_root_test_t4 = 'xxxx/Night_rainy/VOC2007/'
```

## How to use

  1. Please download the trained model from the link in [Google Drive](https://drive.google.com/drive/folders/1_ehe4aIbvIbw38d7PvWDj_fzgZZUF0HE?usp=drive_link)
  2. Test the model by running the following script:

```
  python tools/test.py configs/_project/faster_rcnn_ours.py checkpoint/xxx.pth --eval mAP
```

