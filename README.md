# Automated Upper Tract Urothelial Carcinoma Tumor Segmentation During Ureteroscopy Using Computer Vision Techniques

___
Daiwei Lu, Amy Reed, Natalie Pace, Amy N. Luckenbaugh, Maximilian Pallauf, Nirmish Singla, Ipek Oguz, Nicholas Kavoussi

Journal of Endourology 2024

___
## Install & Requirements
This project should be operated in a conda environment. Otherwise, you will run into a slew of problems, particularly with OpenCV. 

Required install commands: 
- conda install -c conda-forge opencv 

- conda install pytorch torchvision torchaudio -c pytorch
     - *You should go to the [PyTorch website](https://pytorch.org) and perform the generated install command for conda on your machine.*

- conda install -c conda-forge seaborn

- conda install -c conda-forge pandas

- pip install comet_ml

- conda install -c conda-forge tensorboard 

- conda install -c conda-forge scikit-learn

- pip install tqdm

- pip install scikit-image

- pip install segmentation-models-pytorch

- pip install albumentations

___ 
### Public Models

Our public trained models described in the paper can be found [here](). We typically place them in the checkpoints folder and specify their location using -f [location] during testing.


___

### Data

Our dataset is private, but you can load your own. The file structure should be:
```
data
| -> inputs
| | -> train
| | | -> video_folder_1
| | | | -> frame1.jpg
| | | | -> ...
| | -> test
| | | -> video_folder_2
| | | | -> frame1.jpg
| | | | -> ...
| -> labels
| | -> train
| | | -> video_folder1.json
| | | -> ...
| | -> test
| | | -> video_folder2.json
| | | -> ...


```
___
### Comet Logging

Our scripts log training data and results using Comet. Please create your own account and enter the api information for each project in train.py and test.py.

___

### Scripts
1. Run train.py, test.py, and synthesize.py from the slurm directory like so: 
```
python ../<phase>.py [--<argument name> <arg value> ...]
```

Saved models from training will be saved under ./checkpoints/[name]/cp/

```
python train.py --net unet --attn --name unet_061623_attn_fo -d checkpoints -e 100 -q 1 --batch_size 16

python test.py --net unet --attn --name unet_061623_attn_fo_test -f ./checkpoints/unet_061623_attn_fo/cp/unet_061623_attn_fo_CP_epoch45.pth -d checkpoints --batch_size 16
```

___

## Troubleshooting
- The model is highly sensitive to normalization, and tends to perform worse when the training images are normalized. However, **if the model trains on normalized inputs, then inputs for testing and synthesis must also be normalized for the model.** 


