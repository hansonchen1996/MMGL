# Self-Supervised Modality-Aware Multiple Granularity Pre-Training for RGB-Infrared Person Re-Identification

![](pipeline.jpg)

This is the offical implementation of the paper 'Self-Supervised Modality-Aware Multiple Granularity Pre-Training for RGB-Infrared Person Re-Identification'.

# Abstract

RGB-Infrared person re-identification (RGB-IR ReID) aims to associate people across disjoint RGB and IR camera views. Currently, state-of-the-art performance of RGB-IR ReID is not as impressive as that of conventional ReID. Much of that is due to the notorious modality bias training issue brought by the single-modality ImageNet pre-training, which might yield RGB-biased representations that severely hinder the cross-modality image retrieval. This paper makes first attempt to tackle the task from a pre-training perspective. We propose a self-supervised pre-training solution, named Modality-Aware Multiple Granularity Learning (MMGL), which directly trains models from scratch only on multi-modal ReID datasets, but achieving competitive results against ImageNet pre-training, without using any external data or sophisticated tuning tricks. First, we develop a simple-but-effective 'permutation recovery' pretext task that globally maps shuffled RGB-IR images into a shared latent permutation space, providing modality-invariant global representations for downstream ReID tasks. Second, we present a part-aware cycle-contrastive (PCC) learning strategy that utilizes cross-modality cycle-consistency to maximize agreement between semantically similar RGB-IR image patches. This enables contrastive learning for the unpaired multi-modal scenarios, further improving the discriminability of local features without laborious instance augmentation. Based on these designs, MMGL effectively alleviates the modality bias training problem. Extensive experiments demonstrate that it learns better representations (+8.03% Rank-1 accuracy) with faster training speed (converge only in few hours) and higher data efficiency (<5% data size) than ImageNet pre-training. The results also suggest it generalizes well to various existing models, losses and has promising transferability across datasets.

# To Do List

  - [ ] Provide the source code (The code is coming soon!)
  - [ ] Provide models and logs
  - [ ] Stay tuned

## 1. Datasets

- (1) RegDB: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

    - A private download link can be provided via sending me an email (chenyehansen@gmail.com). 
  
- (2) SYSU-MM01: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to preprocess the dataset, the training data will be stored in ".npy" format.
