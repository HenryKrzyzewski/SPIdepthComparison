![image](https://github.com/user-attachments/assets/9a49b07f-af45-419e-aa4b-b3763c5c2d31)Sample outputs on new inputs from the Middlebury stereo vision set. I only used the left image from each stereo set, and these images were NOT included as part of the training data for SPIdepth...

Original images...
![image](https://github.com/user-attachments/assets/c3c24a96-215b-499c-9a8c-301c41bacc60)
![image](https://github.com/user-attachments/assets/6ce11f1f-3595-4a45-9524-fedd1bed5639)
![image](https://github.com/user-attachments/assets/99f496c6-726a-4e83-b8a6-30e677a6be97)

Depth images using SPIdepth...
![image](https://github.com/user-attachments/assets/e05dbf66-48ce-4c48-b9ff-434efe3716a4)
![image](https://github.com/user-attachments/assets/3dc3bd92-da31-480c-baa0-8f718c8b7a3c)
![Uploading image.png…]()

Below is the original README...

# [CVPR 2025] SPIdepth: Strengthened Pose Information for Self-supervised Monocular Depth Estimation

</a> <a href='https://arxiv.org/abs/2404.12501'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=spidepth-strengthened-pose-information-for)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/monocular-depth-estimation-on-kitti-eigen-1)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1?p=spidepth-strengthened-pose-information-for)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/unsupervised-monocular-depth-estimation-on)](https://paperswithcode.com/sota/unsupervised-monocular-depth-estimation-on?p=spidepth-strengthened-pose-information-for)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/monocular-depth-estimation-on-make3d)](https://paperswithcode.com/sota/monocular-depth-estimation-on-make3d?p=spidepth-strengthened-pose-information-for)

## Training

To train on KITTI, run:

```bash
python train.py ./args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```
For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To finetune on KITTI, run:

```bash
python ./finetune/train_ft_SQLdepth.py ./conf/cvnXt.txt ./finetune/txt_args/train/inc_kitti.txt
```

To train on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_train.txt
```
To finetune on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_finetune.txt
```

For preparing cityscapes dataset, please refer to SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

## Pretrained weights and evaluation

You can download weights for some pretrained models here:

* [KITTI](https://huggingface.co/MykolaL/SPIdepth/tree/main/kitti)
* [CityScapes](https://huggingface.co/MykolaL/SPIdepth/tree/main/cityscapes)

To evaluate a model on KITTI, run:

```bash
python evaluate_depth_config.py args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

And to evaluate a model on Cityscapes, run:

```bash
python ./tools/evaluate_depth_cityscapes_config.py args_files/args_cvnXt_H_cityscapes_finetune_eval.txt
```

The ground truth depth files can be found at [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip),
Download this and unzip into `splits/cityscapes`.

## Inference with your own images
```bash
python test_simple_SQL_config.py ./conf/cvnXt.txt
```
In `./conf/cvnXt.txt`, you can set `--image_path` to a single image or a directory of images.

## Citation
If you find this project useful for your research, please consider citing:
~~~
@article{lavreniuk2024spidepth,
      title={SPIdepth: Strengthened Pose Information for Self-supervised Monocular Depth Estimation}, 
      author={Mykola Lavreniuk},
      year={2024},
      journal={arXiv preprint arXiv:2404.12501}
}
~~~
## Acknowledgement
This project is built on top of [SQLdepth](https://github.com/hisfog/SfMNeXt-Impl), and we are grateful for their outstanding contributions.
