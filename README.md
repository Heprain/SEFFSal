# <p align=left>`A SALIENCY ENHANCED FEATURE FUSION BASED MULTISCALE RGB-D SALIENT OBJECT DETECTION NETWORK (ICASSP 2024)`</p>

<!--  -->

This official repository contains the source code, pre-trained, trained checkpoints, and evaluation toolbox of paper 'A Saliency Enhanced Feature Fusion based multiscale RGB-D Salient Object Detection Network'. The technical report could be found at [arXiv]().

We invite all to contribute in making it more acessible and useful. If you have any questions about our work, feel free to contact me via e-mail (qingy_zhao@163.com). 

<p align="center">
    <img src="pics/fm.pdf" width="600"  width="1200"/> <br />
    <em> 
    Figure 1: Comparisons between the existing methods and our DFormer (RGB-D Pre-training).
    </em>
</p>

<!-- <p align="center">
    <img src="figs/overview.jpg" width="600"  width="1200"/> <br />
    <em> 
    Figure 2: Overview of the DFormer.
    </em>
</p> -->






## 1. 🌟  NEWS 

<!-- - [2023/09/05] Releasing the codebase of DFormer and all the pre-trained checkpoints.

- [2023/10/26] Releasing the RGBD SOD codebase of DFormer at [DFormer-SOD](https://github.com/VCIP-RGBD/DFormer-SOD).

- [2023/12/03] Adding the tutorial about adding new datasets at [Application to new datasets(添加新数据集)](https://github.com/VCIP-RGBD/DFormer/tree/main/figs/application_new_dataset). -->

- [2024/01/16] Our DFormer has been accpeted by The International Conference on Learning Representations (ICLR 2024).


## 2. 🚀 Get Start

**0. Install**

```
conda create -n dformer python=3.10 -y
conda activate dformer
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install tqdm opencv-python scipy tensorboardX tabulate easydict
```


**1. Download Datasets and Checkpoints.**



- **Datasets:** 

By default, you can put datasets into the folder 'datasets' or use 'ln -s path_to_data datasets'.

| Datasets | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|
- **Checkpoints:** 

ImageNet-1K Pre-trained DFormers T/S/B/L and NYUDepth or SUNRGBD trained DFormers T/S/B/L can be downloaded at:
<!-- 
| Pre-trained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL) | [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | 
|:---: |:---:|:---:|:---:|



NYUDepth v2 trained DFormers T/S/B/L can be downloaded at 

| NYUDepth v2 | [GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf) | [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | 
|:---: |:---:|:---:|:---:|


*SUNRGBD 

| SUNRGBD | [GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che) | [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | 
|:---: |:---:|:---:|:---:| -->


| Weights | GoogleDrive | OneDrive | BaiduNetdisk|
|-------|-------| - | - |
| Pretrained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL) | [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | 
|NYUDepthv2 (57.2mIoU)|[GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf) | [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | 
|SUNRGBD (52.5mIoU)|[GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che) | [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | 


 <br />


<details>
<summary>Orgnize the checkpoints and dataset folder in the following structure:</summary>
<pre><code>

```shell
<checkpoints>
|-- <pretrained>
    |-- <DFormer_Large.pth.tar>
    |-- <DFormer_Base.pth.tar>
    |-- <DFormer_Small.pth.tar>
    |-- <DFormer_Tiny.pth.tar>
|-- <trained>
    |-- <NYUDepthv2>
        |-- ...
    |-- <SUNRGBD>
        |-- ...
<datasets>
|-- <DatasetName1>
    |-- <RGB>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <Depth>
        |-- <name1>.<DepthFormat>
        |-- <name2>.<DepthFormat>
    |-- train.txt
    |-- test.txt
|-- <DatasetName2>
|-- ...
```

</code></pre>
</details>




 <br /> 




**2. Train.**

You can change the `local_config' files in the script to choose the model for training. 
```
bash train.sh
```

After training, the checkpoints will be saved in the path `checkpoints/XXX', where the XXX is depends on the training config.


**3. Eval.**

You can change the `local_config' files and checkpoint path in the script to choose the model for testing. 
```
bash eval.sh
```

**4. Visualize.**

```
bash infer.sh
```


## 🚩 Performance

<p align="center">
    <img src="figs/Semseg.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p>

<!-- <p align="center">
    <img src="figs/Sal.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p> -->

## 🕙 ToDo
- [ ] Tutorial on applying the DFormer encoder to the frameworks of other tasks
- [ ] Release the code of RGB-D pre-training.
- ~~[-] Tutorial on applying to a new dataset.~~
- ~~[-] Release the DFormer code for RGB-D salient obejct detection.~~

> We invite all to contribute in making it more acessible and useful. If you have any questions or suggestions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn) or raise an issue. 


## Reference
You may want to cite:
```
@article{yin2023dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhongyu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2309.09668},
  year={2023}
}
```


### Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [CMNext](https://github.com/jamycheung/DELIVER). Thanks for their authors.



### License

Code in this repo is for non-commercial use only.
