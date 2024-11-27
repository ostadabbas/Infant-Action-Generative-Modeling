# Infant Action Generative Modeling

Codes and experiments for the following paper: 

Xiaofei Huang, Elaheh Hatami, Amal Mathew, Sarah Ostadabbas, “Infant Action Generative Modeling” [WACV 2025 ACCEPTED]

Contact: 

[Xiaofei Huang](xhuang@ece.neu.edu)
[Elaheh Hatami](e.hatamimajoumerd@northeastern.edu )
[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

## Table of Contents
  * [Introduction](#introduction)
  * [Environment](#environment)
  * [How To Use](#how-to-use)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)

## Introduction
This is an official pytorch implementation of [*Infant Action Generative Modeling*](LINK). We present a pioneering infant action generation and classification (InfAGenC) pipeline. This transformative approach combines a transformer-based variational autoencoder (VAE) with a spatial-temporal graph convolutional network (ST-GCN) to produce synthetic infant action samples. By iteratively refining the generative model with diverse yet accurate data, we enhance the realism of synthetic data, resulting in more precise infant action recognition models.   </br>

##### Generated Examples Based on InfActPrimitive Data

![infgrid](images/primitive_fig_200_grid.gif#gh-light-mode-only)

##### Generated Examples Based on InfantAction Data

![infgrid](images/transitional_fig_200.gif#gh-light-mode-only)


## Environment
The code is developed using python 3.8 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA TITAN Xp GPU card. Other platforms or GPU cards are not fully tested.

## How To Use
### Create conda environment

```
conda env create -f environment.yml
conda activate infant_generation
```

### Body Model preparation
```bash
bash prepare/download_smpl_files.sh
```

This will create the folder ``${GEN_ROOT}/models/smpl`` and download the SMPL neutral model from this [github repo](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl) and related files under it.

As working on infant case, you must also download the SMIL model from the [SMIL website](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html) and place 'smil_web.pkl' file into ``${GEN_ROOT}/models/smpl``. Then change the ``SMPL_MODEL_PATH`` variable in ``${GEN_ROOT}/src/config.py`` to refer the SMIL model.

### Data preparation

For the InfActPrimitive data for the infant generative model and action recognition model training, we have preprocessed them and split into train, validate, and test sets. Please download from [Processed_InfActPrimitive dataset](https://coe.northeastern.edu/Research/AClab/Processed_InfActPrimitive/), extract and place them under ``${GEN_ROOT}/data/InfActPrimitive``, and make them look like this:

   ```
   ${GEN_ROOT}
    `-- data
        `-- InfActPrimitive
            |-- README.md
            `-- Data
                |-- primitive_train.pkl
                |-- primitive_validate.pkl
                `-- primitive_test.pkl
   ```


For the InfantAction data for the infant generative model and action recognition model training, we have preprocessed them and split into train and test sets. Please download from [Processed_InfantAction dataset](https://coe.northeastern.edu/Research/AClab/InfantAction/), extract and place them under ``${GEN_ROOT}/data/InfActPlus``, and make them look like this:

   ```
   ${GEN_ROOT}
    `-- data
        `-- InfActPlus
            |-- train_trans_data.pkl
            |-- test_trans_data.pkl
            `-- README.md
   ```


### Initializing action recognition model
First train a ST-GCN model on real training set. 

For InfActPrimitive data, there are 5 action classes: Supine, Prone, Sitting, Standing, All-fours. 
```
python -m src.train.train_stgcn --dataset infactposture --extraction_method vibe --pose_rep rot6d --num_epochs 15 --snapshot 5 --batch_size 16 --lr 0.0001 --num_frames 60 --view all --sampling conseq --num_classes 5 --sampling_step 1 --glob --no-translation --folder exps/pretrained_recognition_15epochs
```

For InfantAction data, there are 4 action classes: Crawling, Sitting, Standing, Rolling. 
```
python -m src.train.train_stgcn --dataset infacttrans --extraction_method vibe --pose_rep rot6d --num_epochs 15 --snapshot 5 --batch_size 16 --lr 0.0001 --num_frames 90 --view all --sampling conseq --num_classes 5 --sampling_step 1 --glob --no-translation --folder exps/pretrained_recognition_15epochs
```

### Initializing action generative model
Pretrain a pure VAE model on real training set. 

For InfActPrimitive data
```
python -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl_velxyz --pose_rep rot6d --lambda_velxyz 1e-3 --lambda_kl 1e-5 --jointstype smpl --batch_size 16 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-translation --no-vertstrans --dataset infactposture --num_classes 5 --num_epochs 1100 --snapshot 20 --folder exps/infactposture_rc_rcxyz_velxyz_epoch1100
```

For InfantAction data
```
python -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl_velxyz --pose_rep rot6d --lambda_velxyz 1e-3 --lambda_kl 1e-5 --jointstype smpl --batch_size 16 --num_frames 90 --num_layers 8 --lr 0.0001 --glob --no-translation --no-vertstrans --dataset infacttrans --num_classes45 --num_epochs 1100 --snapshot 20 --folder exps/infacttrans_rc_rcxyz_velxyz_epoch1100
```


### Training InfAGenC model
Based on the pretrained recognition and generation models, continue to enhance the generative model by introducing generated motions into training set.
Please modify the augments `--init_recognition` and  `--init_generation` to load pretrained action recognition model and action generation model.

For InfActPrimitive data
```
python -m src.train.train_cvae_stgcn --modelname cvae_transformer_rc_rcxyz_kl_velxyz --pose_rep rot6d --lambda_velxyz 1e-3 --lambda_kl 1e-5 --jointstype smpl --batch_size 16 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-translation --no-vertstrans --dataset infactposture --num_classes 5 --num_epochs 200 --snapshot 20 --init_recognition PATH/TO/INITIALIZED RECOGNITION MODEL CHECKPOINT.tar --init_generation PATH/TO/INITIALIZED GENERATION MODEL CHECKPOINT.tar --folder exps/infactposture_rc_rcxyz_velxyz_init1100_epoch200
```

For InfantAction data
```
python -m src.train.train_cvae_stgcn --modelname cvae_transformer_rc_rcxyz_kl_velxyz --pose_rep rot6d --lambda_velxyz 1e-3 --lambda_kl 1e-5 --jointstype smpl --batch_size 16 --num_frames 90 --num_layers 8 --lr 0.0001 --glob --no-translation --no-vertstrans --dataset infacttrans --num_classes 5 --num_epochs 200 --snapshot 20 --init_recognition PATH/TO/INITIALIZED RECOGNITION MODEL CHECKPOINT.tar --init_generation PATH/TO/INITIALIZED GENERATION MODEL CHECKPOINT.tar --folder exps/infacttrans_rc_rcxyz_velxyz_init1100_epoch200
```

After InfAGenC training, high-quality synthetic samples generated. They can be used for infant action recognition task.

### Pretrained models
You can download pretrained models and save in ``${GEN_ROOT}/exps``:

For InfActPrimitive data
1. [Pretrained action recognition model](https://drive.google.com/file/d/1c5Crbw-l1LRCYVl9xeHgOsR2MVd4pGax/view?usp=drive_link)

2. [Pretrained action generation model](https://drive.google.com/file/d/14DFo36-4UhS-_jdXrEgjjmF1iu213DZi/view?usp=drive_link)

3. [Trained InfAGenC model](https://drive.google.com/file/d/12TnFIY0eM0pJGI2CfTwuYmc6DvWloiUY/view?usp=drive_link)

For InfantAction data
1. [Pretrained action recognition model](https://drive.google.com/file/d/1BOEOosxjLtsFljVu8n-Vtfa-D8ceuCF2/view?usp=drive_link)

2. [Pretrained action generation model](https://drive.google.com/file/d/1xOQ9WHuLQuImJ2zDoAH9ZU32-QCe4JlO/view?usp=drive_link)

3. [Trained InfAGenC model](https://drive.google.com/file/d/1iVpS6_Mb8HgLXDTdq0fZQMSA0dSNqWEO/view?usp=drive_link)

 
### Generation
Please specify the checkpoint of your trained InfAGenC model.
```
python -m src.generate.generate_sequences PATH/TO/GENERATION MODEL CHECKPOINT.tar --num_samples_per_action 50 --cpu
```

### Visualization
Please specify the checkpoint of your trained InfAGenC model and also the number of classes.
```
python -m src.visualize.visualize_checkpoint PATH/TO/GENERATION MODEL CHECKPOINT.tar --num_actions_to_sample NUMBER_OF_CLASSES  --num_samples_per_action 5
```

### Training infant action recognition model
Once you generated synthetic samples during InfAGenC model training, they can be mixed with real data for infant action recognition model training.

For InfActPrimitive data
```
python -m src.train.train_stgcn --dataset mixposture --extraction_method vibe --pose_rep rot6d --num_epochs 100 --snapshot 20 --batch_size 16 --lr 0.0001 --num_frames 60 --view all --sampling conseq --num_classes 5 --sampling_step 1 --glob --no-translation --folder
```
If you want to use our generated samples, please download [here](https://drive.google.com/file/d/12WTnGlZyBFAAahB7QxB5boIF69STWjAs/view?usp=drive_link) and save in ``${GEN_ROOT}/exps/infactposture_rc_rcxyz_velxyz_init1100_epoch200``

For InfantAction data
```
python -m src.train.train_stgcn --dataset mixtrans --extraction_method vibe --pose_rep rot6d --num_epochs 100 --snapshot 20 --batch_size 16 --lr 0.0001 --num_frames 90 --view all --sampling conseq --num_classes 4 --sampling_step 1 --glob --no-translation --folder
```
If you want to use our generated samples, please download [here](https://drive.google.com/file/d/19Q6UiDQEAQY9dvZpJ1NOhh9mUJJP5Qt6/view?usp=drive_link) and and save in ``${GEN_ROOT}/exps/infacttrans_rc_rcxyz_velxyz_init1100_epoch200``



## Citation

If you use our code or models in your research, please cite with:

```
@inproceedings{Huang2025InfantAction,
  title={Infant Action Generative Modeling},
  author={Huang, Xiaofei and Hatamimajoumerd, Elaheh and Mathew, Amal and Ostadabbas, Sarah},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}

```

## Acknowledgement
Thanks for the open-source ACTOR

* [ACTOR: Action-Conditioned 3D Human Motion Synthesis with Transformer VAE, Petrovich, Mathis and Black, Michael J. and Varol, G](https://github.com/Mathux/ACTOR)

Thanks for the InfActPrimitive dataset

* [Challenges in Video-Based Infant Action Recognition: A Critical Examination of the State of the Art, Hatamimajoumerd E, Daneshvar KP, Huang X, Luan L, Amraee Somaieh, Ostadabbas S](https://github.com/ostadabbas/Video-Based-Infant-Action-Recognition)

## License 
* This code is for non-commertial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 


Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.


