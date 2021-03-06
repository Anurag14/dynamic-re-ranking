**Related Project:** [Strong Identification Loss Baseline](https://github.com/huanghoujing/beyond-part-models)

This project provides a strong triplet loss baseline in person re-identification, using [pytorch](https://github.com/pytorch/pytorch).


# Current Results

Triplet loss with settings:
- ResNet-50, `stride = 2` or `stride = 1` in last conv block
- NOT normalizing feature to unit length, with margin 0.3
- Only horizontal flipping used for data augmentation
- `im_w x im_h = 128 x 256`, `ims_per_id = 4`, `ids_per_batch = 32`
- Adam optimizer, base learning rate 2e-4, decaying exponentially after 150 epochs. Train for 300 epochs in total.

The results are as follows. `S1` and `S2` means `stride = 1` and `stride = 2` respectively; `R.R.` means using re-ranking.

|               | Rank-1 (%) | mAP (%) | R.R. Rank-1 (%) | R.R. mAP (%) |
| ---           | :---: | :---: | :---: | :---: |
| Market1501-S2 | 86.43 | 71.50 | 89.82 | 85.55 |
| Market1501-S1 | 89.04 | 75.29 | 91.57 | 87.82 |
| Duke-S2       | 78.82 | 61.09 | 83.98 | 79.15 |
| Duke-S1       | 79.76 | 64.27 | 85.32 | 81.48 |
| CUHK03-S2     | 56.36 | 50.82 | 65.21 | 65.76 |
| CUHK03-S1     | 59.14 | 54.43 | 69.86 | 70.03 |

**We see that `stride = 1` (higher spatial resolution before global pooling) has obvious improvement over `stride = 2` (original ResNet). I tried this inspired by paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349).**


Other details of setting can be found in the code. To test my trained models or reproduce these results, see the [Examples](#examples) section.


# Resources

This repository contains following resources

- A beginner-level dataset interface independent of Pytorch, Tensorflow, etc, supporting multi-thread prefetching (README file is under way)
- Three most used ReID datasets, Market1501, CUHK03 (new protocol) and DukeMTMC-reID
- Python version ReID evaluation code (Originally from [open-reid](https://github.com/Cysu/open-reid))
- Python version Re-ranking (Originally from [re_ranking](https://github.com/zhunzhong07/person-re-ranking/blob/master/python-version/re_ranking))
- Triplet Loss


# Installation

It's recommended that you create and enter a python virtual environment, if versions of the packages required here conflict with yours.

I use Python 2.7 and Pytorch 0.3. For installing Pytorch, follow the [official guide](http://pytorch.org/). Other packages are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Then clone the repository:

```bash
git clone https://github.com/huanghoujing/person-reid-triplet-loss-baseline.git
cd person-reid-triplet-loss-baseline
```


# Dataset Preparation

Inspired by Tong Xiao's [open-reid](https://github.com/Cysu/open-reid) project, dataset directories are refactored to support a unified dataset interface.

Transformed dataset has following features
- All used images, including training and testing images, are inside the same folder named `images`
- Images are renamed, with the name mapping from original images to new ones provided in a file named `ori_to_new_im_name.pkl`. The mapping may be needed in some cases.
- The train/val/test partitions are recorded in a file named `partitions.pkl` which is a dict with the following keys
  - `'trainval_im_names'`
  - `'trainval_ids2labels'`
  - `'train_im_names'`
  - `'train_ids2labels'`
  - `'val_im_names'`
  - `'val_marks'`
  - `'test_im_names'`
  - `'test_marks'`
- Validation set consists of 100 persons (configurable during transforming dataset) unseen in training set, and validation follows the same ranking protocol of testing.
- Each val or test image is accompanied by a mark denoting whether it is from
  - query (`mark == 0`), or
  - gallery (`mark == 1`), or
  - multi query (`mark == 2`) set

## Market1501

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4) or [BaiduYun](https://pan.baidu.com/s/1nvOhpot). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

## CUHK03

We follow the new training/testing protocol proposed in paper
```
@article{zhong2017re,
  title={Re-ranking Person Re-identification with k-reciprocal Encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={CVPR},
  year={2017}
}
```
Details of the new protocol can be found [here](https://github.com/zhunzhong07/person-re-ranking).

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1Ssp9r4g8UbGveX-9JvHmjpcesvw90xIF) or [BaiduYun](https://pan.baidu.com/s/1hsB0pIc). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the CUHK03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). Then download the training/testing partition file from [Google Drive](https://drive.google.com/open?id=14lEiUlQDdsoroo8XJvQ3nLZDIDeEizlP) or [BaiduYun](https://pan.baidu.com/s/1miuxl3q). This partition file specifies which images are in training, query or gallery set. Finally run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_cuhk03.py \
--zip_file ~/Dataset/cuhk03/cuhk03_release.zip \
--train_test_partition_file ~/Dataset/cuhk03/re_ranking_train_test_split.pkl \
--save_dir ~/Dataset/cuhk03
```


## DukeMTMC-reID

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1P9Jr0en0HBu_cZ7txrb2ZA_dI36wzXbS) or [BaiduYun](https://pan.baidu.com/s/1miIdEek). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```


## Combining Trainval Set of Market1501, CUHK03, DukeMTMC-reID

Larger training set tends to benefit deep learning models, so I combine trainval set of three datasets Market1501, CUHK03 and DukeMTMC-reID. After training on the combined trainval set, the model can be tested on three test sets as usual.

Transform three separate datasets as introduced above if you have not done it.

For the trainval set, you can download what I have transformed from [Google Drive](https://drive.google.com/open?id=1hmZIRkaLvLb_lA1CcC4uGxmA4ppxPinj) or [BaiduYun](https://pan.baidu.com/s/1jIvNYPg). Otherwise, you can run the following script to combine the trainval sets, replacing the paths with yours.

```bash
python script/dataset/combine_trainval_sets.py \
--market1501_im_dir ~/Dataset/market1501/images \
--market1501_partition_file ~/Dataset/market1501/partitions.pkl \
--cuhk03_im_dir ~/Dataset/cuhk03/detected/images \
--cuhk03_partition_file ~/Dataset/cuhk03/detected/partitions.pkl \
--duke_im_dir ~/Dataset/duke/images \
--duke_partition_file ~/Dataset/duke/partitions.pkl \
--save_dir ~/Dataset/market1501_cuhk03_duke
```

## Configure Dataset Path

The project requires you to configure the dataset paths. In `tri_loss/dataset/__init__.py`, modify the following snippet according to your saving paths used in preparing datasets.

```python
# In file tri_loss/dataset/__init__.py

########################################
# Specify Directory and Partition File #
########################################

if name == 'market1501':
  im_dir = ospeu('~/Dataset/market1501/images')
  partition_file = ospeu('~/Dataset/market1501/partitions.pkl')

elif name == 'cuhk03':
  im_type = ['detected', 'labeled'][0]
  im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
  partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))

elif name == 'duke':
  im_dir = ospeu('~/Dataset/duke/images')
  partition_file = ospeu('~/Dataset/duke/partitions.pkl')

elif name == 'combined':
  assert part in ['trainval'], \
    "Only trainval part of the combined dataset is available now."
  im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
  partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')
```

## Evaluation Protocol

Datasets used in this project all follow the standard evaluation protocol of Market1501, using CMC and mAP metric. According to [open-reid](https://github.com/Cysu/open-reid), the setting of CMC is as follows

```python
# In file tri_loss/dataset/__init__.py

cmc_kwargs = dict(separate_camera_set=False,
                  single_gallery_shot=False,
                  first_match_break=True)
```

To play with [different CMC options](https://cysu.github.io/open-reid/notes/evaluation_metrics.html), you can [modify it accordingly](https://github.com/Cysu/open-reid/blob/3293ca79a07ebee7f995ce647aafa7df755207b8/reid/evaluators.py#L85-L95).

```python
# In open-reid's reid/evaluators.py

# Compute all kinds of CMC scores
cmc_configs = {
  'allshots': dict(separate_camera_set=False,
                   single_gallery_shot=False,
                   first_match_break=False),
  'cuhk03': dict(separate_camera_set=True,
                 single_gallery_shot=True,
                 first_match_break=False),
  'market1501': dict(separate_camera_set=False,
                     single_gallery_shot=False,
                     first_match_break=True)}
```


# Examples

## Inference

You can use a trained model to extract features for a list of images, and then perform whatever you desire with these features. An example is
```bash
python script/experiment/infer_images_example.py \
--model_weight_file YOUR_MODEL_WEIGHT_FILE
```

## Test

My training log and saved model weights for three datasets can be downloaded from [Google Drive](https://drive.google.com/open?id=14ljnClpZkHD7BzrET1q1eFQ_XhaRzM3-) or [BaiduYun](https://pan.baidu.com/s/1mjfTcxy).

Specify
- a dataset name (one of `market1501`, `cuhk03`, `duke`)
- stride, `1` or `2`
- an experiment directory for saving testing log
- the path of the downloaded `model_weight.pth`

in the following command and run it.

```bash
python script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset DATASET_NAME \
--last_conv_stride STRIDE \
--normalize_feature false \
--exp_dir EXPERIMENT_DIRECTORY \
--model_weight_file THE_DOWNLOADED_MODEL_WEIGHT_FILE
```

## Train

You can also train it by yourself. The following command performs training, validation and finally testing automatically.

Specify
- a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
- stride, `1` or `2`
- training on `trainval` set or `train` set (for tuning parameters)
- an experiment directory for saving training log

in the following command and run it.

```bash
python script/experiment/train.py \
-d '(0,)' \
--only_test false \
--dataset DATASET_NAME \
--last_conv_stride STRIDE \
--normalize_feature false \
--trainset_part TRAINVAL_OR_TRAIN \
--exp_dir EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5
```

### Log

During training, you can run the [TensorBoard](https://github.com/lanpa/tensorboard-pytorch) and access port `6006` to watch the loss curves etc. E.g.

```bash
# Modify the path for `--logdir` accordingly.
tensorboard --logdir YOUR_EXPERIMENT_DIRECTORY/tensorboard
```

For more usage of TensorBoard, see the website and the help:

```bash
tensorboard --help
```


## Visualize Ranking List

Specify
- a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
- stride, `1` or `2`
- either `model_weight_file` (the downloaded `model_weight.pth`) OR `ckpt_file` (saved `ckpt.pth` during training)
- an experiment directory for saving images and log

in the following command and run it.

```bash
python script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 16 \
--rank_list_size 10 \
--dataset DATASET_NAME \
--last_conv_stride STRIDE \
--normalize_feature false \
--exp_dir EXPERIMENT_DIRECTORY \
--model_weight_file '' \
--ckpt_file ''
```

Each query image and its ranking list would be saved to an image in directory `EXPERIMENT_DIRECTORY/rank_lists`. As shown in following examples, green boundary is added to true positive, and red to false positve.

![](example_rank_lists_on_Duke/00000126_0002_00000021.jpg)

![](example_rank_lists_on_Duke/00000147_0003_00000004.jpg)

![](example_rank_lists_on_Duke/00000169_0001_00000008.jpg)

![](example_rank_lists_on_Duke/00000257_0003_00000004.jpg)


# Time and Space Consumption


Test with CentOS 7, Intel(R) Xeon(R) CPU E5-2618L v3 @ 2.30GHz, GeForce GTX TITAN X.

**Note that the following time consumption is not gauranteed across machines, especially when the system is busy.**

### GPU Consumption in Training

For following settings
- ResNet-50, `stride=1` in last block
- `identities_per_batch = 32`, `images_per_identity = 4`, `images_per_batch = 32 x 4 = 128`
- image size `h x w = 256 x 128`

it occupies ~11000MB GPU memory.

If not having a 12 GB GPU, you have to either decrease `identities_per_batch` or use multiple GPUs.


### Training Time

Taking Market1501 as an example, it contains `31969` training images of `751` identities, thus `1 epoch = 751 / 32 = 24 iterations`. Each iteration takes ~1.08s, so each epoch ~27s. Training for 300 epochs takes ~2.25 hours.

### Testing Time

Taking Market1501 as an example
- With `images_per_batch = 32`, extracting feature of whole test set (12936 images) takes ~160s.
- Computing query-gallery global distance, the result is a `3368 x 15913` matrix, ~2s
- Computing CMC and mAP scores, ~15s
- Re-ranking requires computing query-query distance (a `3368 x 3368` matrix) and gallery-gallery distance (a `15913 x 15913` matrix, most time-consuming), ~90s


# References & Credits

- [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
- [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349)
- [open-reid](https://github.com/Cysu/open-reid)
- [Re-ranking Person Re-identification with k-reciprocal Encoding](https://github.com/zhunzhong07/person-re-ranking)
- [Market1501](http://www.liangzheng.org/Project/project_reid.html)
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation)
