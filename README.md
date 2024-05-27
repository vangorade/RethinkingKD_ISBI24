
The code repository of IEEE ISBI 2024 (Oral!) paper [RETHINKING INTERMEDIATE LAYERS DESIGN IN KNOWLEDGE DISTILLATION FOR KIDNEY AND LIVER TUMOR SEGMENTATION](https://arxiv.org/pdf/2311.16700.pdf)

# Structure of this repository
This repository is organized as:
- [data](/data/) preprocessed data
- [datasets](/datasets/) dataloader for different datasets
- [networks](/networks/) models code
- [scripts](/scrips/) scripts for preparing data
- [utils](/utils/) training and processing data
- [train.py](/train.py) train a teacher model
- [train_kd.py](/train_kd.py) train with knwoledge distillation

# Usage Guide

## Dataset Preparation

### KiTS

Preprocessed KiTS19 data is available [here](https://www.dropbox.com/scl/fi/7jrh5ufzxonwj8mswoa5p/kits19.tar.gz?rlkey=gsadmf861vq9wy1h2qni4iyr7&st=9sqc1nlt&dl=0)

Or, you can follow instructions below to preprocess your own data.

Download data [here](https://github.com/neheller/kits19)

Please follow the instructions and the data/ directory should then be structured as follows
```
data
├── case_00000
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
├── case_00001
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
...
├── case_00209
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
└── kits.json
```
Cut 3D data into slices using ```scripts/SliceMaker.py```. You can also check ```Prep.ipynb```. 

```
python scripts/SliceMaker.py --inpath /data/kits19/data --outpath /data/kits/train --dataset kits --task tumor
```
Process is similar for any other dataset such as KiTS, ACDC, etc..

## Running
### Training Teacher Model
Before knowledge distillation, a well-trained teacher model is required. ```/train.py``` is used to trained a single model without KD(either a teacher model or a student model). 

```
python train.py --model raunet --checkpoint_path /data/checkpoints
```

After training, the checkpoints will be stored in ```/data/checkpoints``` as assigned.

If you want to try different models, use ```--model```.

### Training With Knowledge Distillation 
For example, use resnet_18 as student model

```
python train_kd.py --tckpt /data/checkpoints/name_of_teacher_checkpoint.ckpt --smodel resnet_18
```

```--tckpt``` refers to the path of teacher model checkpoint. And you can change student model by revising ```--smodel```

### Acknowledgements

Thanks to [HDCDN](github.com/EagleMIT/Hilbert-Distillation), [SKD](https://github.com/irfanICMLL/structure_knowledge_distillation) [MedCAM](https://pypi.org/project/medcam/) for their wonderfull work.