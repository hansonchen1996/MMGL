# Self-Supervised Modality-Aware Multiple Granularity Pre-Training for RGB-Infrared Person Re-Identification

![](pipeline.png)

This is the offical implementation of the paper 'Self-Supervised Modality-Aware Multiple Granularity Pre-Training for RGB-Infrared Person Re-Identification'.

**Authors**: *Lin Wan, Qianyan Jing, Zongyuan Sun, Chuang Zhang, Zhihang Li, and Yehansen Chen*

# Abstract

RGB-Infrared person re-identification (RGB-IR ReID) aims to associate people across disjoint RGB and IR camera views. Currently, state-of-the-art performance of RGB-IR ReID is not as impressive as that of conventional ReID. Much of that is due to the notorious modality bias training issue brought by the single-modality ImageNet pre-training, which might yield RGB-biased representations that severely hinder the cross-modality image retrieval. This paper makes first attempt to tackle the task from a pre-training perspective. We propose a self-supervised pre-training solution, named Modality-Aware Multiple Granularity Learning (MMGL), which directly trains models from scratch only on multi-modal ReID datasets, but achieving competitive results against ImageNet pre-training, without using any external data or sophisticated tuning tricks. First, we develop a simple-but-effective 'permutation recovery' pretext task that globally maps shuffled RGB-IR images into a shared latent permutation space, providing modality-invariant global representations for downstream ReID tasks. Second, we present a part-aware cycle-contrastive (PCC) learning strategy that utilizes cross-modality cycle-consistency to maximize agreement between semantically similar RGB-IR image patches. This enables contrastive learning for the unpaired multi-modal scenarios, further improving the discriminability of local features without laborious instance augmentation. Based on these designs, MMGL effectively alleviates the modality bias training problem. Extensive experiments demonstrate that it learns better representations (+8.03% Rank-1 accuracy) with faster training speed (converge only in few hours) and higher data efficiency (<5% data size) than ImageNet pre-training. The results also suggest it generalizes well to various existing models, losses and has promising transferability across datasets.

# To Do List

  - [x] Release the fine-tuned models and training logs by MMGL
  - [ ] Release the source code (**The code is coming soon!**)
  - [ ] Release the pre-trained models and logs
  - [ ] Stay tuned

# How to Use

## Environment

**Packages**

- Python 3.6.13
- PyTorch 1.10.2
- Numpy 1.19.2
- Scipy 1.5.2
- TensorboardX 2.2

**Hardware**

- A single Nvidia 2080Ti (original paper) / 3080Ti (what we use now)
- GPU Memory: 12G
- Nvidia Driver Version: 510.54
- CUDA Version: 11.6 

## Datasets

- (1) RegDB: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

  
- (2) SYSU-MM01: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to preprocess the dataset, the training data will be stored in ".npy" format.

- A private download link of both datasets can be provided via sending me an email (chenyehansen@gmail.com). 

## Self-supervised Pre-Training with MMGL

Only **single-gpu** training is supported now.

To do MMGL pre-training on a *two-stream* ResNet-50 backbone, run:
```
python train.py --dataset sysu --stream two --lr 0.1 --gpu 0
```

**Optional Hyper-Parameters：**

`--num_stripe` : The number of partition stripes

`--cl_weight`: The weight of PCC loss

`--cl_temp`: The temperature of PCC loss


To do MMGL pre-training on a *one-stream* ResNet-50 backbone, run:
```
python train.py --dataset sysu --stream one --lr 0.1 --gpu 0
```

**Pre-trained Models：**

Backbone | Training Time | Permutation Accuracy | Model
---|:---:|:---:|:---:
Two-Stream     | 6h  | 98.6% | avaliable soon
One-Stream  | 6h  | 97.5% | avaliable soon


## Supervised RGB-Infrared Person Re-Identification

Once the pre-training is finished, please move it to the corresponding ```save_model/``` dictionary of different methods.

To perform supervised RGB-IR ReID with Base / AGW, run:
```
cd AGW

python train.py --dataset sysu (or regdb) --mode all --lr 0.1 --method agw (or base) --gpu 0 --resume 'wirite your checkpoint file name here'
```

To perform supervised RGB-IR ReID with DDAG, run:
```
cd DDAG

python train_ddag.py --dataset sysu(regdb) --lr 0.1 --wpa --graph --gpu 0 --resume 'wirite your checkpoint file name here'
```

**MMGL Pre-Training Fine-Tuned Results:**
|Methods    | Pretrained| Rank@1  | mAP |  Model|
| --------   | -----    | -----  |  ----- |------|
|AGW  | MMGL | 56.97%   | 54.61%  | [Checkpoint](https://drive.google.com/file/d/1y_GmFSWiVtsu0_Zf5tENLU0BTf6j9qfB/view?usp=sharing) \| [Training Log](https://drive.google.com/file/d/1xSdwuZ6AP3J-8Qi-dOBFw4J723I7m6eS/view?usp=sharing)|
|DDAG     | MMGL | 56.75%  | 53.96% |[Checkpoint](https://drive.google.com/file/d/1hXYVXwfwNdL5JS9BPWvGwGD5ZB3FPzCy/view?usp=sharing) \| [Training Log](https://drive.google.com/file/d/1rpwVqG0q_O-Jg7Yz9itx0VZj4Euxy6GK/view?usp=sharing)|

\* Both of these two methods may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters.

**ImageNet Supervised Pre-Training Fine-Tuned Results (Provided by Mang Ye):**

|Methods    | Pretrained| Rank@1  | mAP  |  Model|
| --------   | -----    | -----  |  -----   |------|
|AGW  | ImageNet | ~ 47.50%  | ~ 47.65% | [Checkpoint](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)|
|DDAG      | ImageNet | ~ 54.75% | ~53.02% |----- |

# Citation

Please cite this paper in your publications if it helps your research:
```
@article{wan2021self,
  title={Self-Supervised Modality-Aware Multiple Granularity Pre-Training for RGB-Infrared Person Re-Identification},
  author={Wan, Lin and Jing, Qianyan and Sun, Zongyuan and Zhang, Chuang and Li, Zhihang and Chen, Yehansen},
  journal={arXiv preprint arXiv:2112.06147},
  year={2021}
}
```
