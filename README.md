# A Multimodal Deep Neural Network for Causally Learning Chest X-ray Images
Causal intervention has been widely used in deep learning to tackle the confounding problems when the data are out of distribution. A representative class of causal-intervention-based strategy is invariant risk minimization, yet manually annotating is indispensable to indicate the environmental splits. A feasible solution is to use a pair of complementary attention, one of which focuses on the foreground target and another extracts the background features. The environments are then split unsupervisedly. However, in terms of medical images, the background is not as simple as normal images (e.g., a camel in desert), and using all the background feature as confounder is redundant and will degrade the expected generalization performance. In this paper, we develop a novel multimodal deep neural network which automatically extracts the confounded background features of chest X-ray (CXR) images by a multimodal approach, i.e., feature fusion with explicit confounders such as demographic information. To achieve this, we design an architecture with two modules, a classifier module taking images as input and an environment split module taking demographic tables as input. The cross attention is then conducted between the encoded background features and demographic features to extract the actual confounded background features. The proposed method meaningfully improves the generalization performance on multiple CXR datasets and accurately locates the lesions. 

## Requirements
- pytorch
- torchvision
- numpy
- scipy
- pandas
- yaml
- timm
- sklearn

## Preprocess Training Set
```shell
python dataset_gen.py
```

## Preprocess Test Set
```shell
python dataset_gen.py --test_set ./chexpert
python dataset_gen.py --test_set ./chestxray8
python dataset_gen.py --test_set ./openi
```

## Normalize the Trainin Set
```shell
python get_mean_std.py
```

## Train
```shell
python train.py
```

## Test
```shell
python test.py
```

## Citation


Please cite our work if you find our code/paper is useful to your work.


```
@INPROCEEDINGS{9995021,

  author={Ma, Qin and Zeng, Lin and Tu, Shikui and Xu, Lei},

  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 

  title={A Multimodal Deep Neural Network for Causally Learning Chest X-ray Images}, 

  year={2024},

  volume={},

  number={},

  pages={509-514},

  doi={10.1109/BIBM55620.2022.9995021}}

```
