# StoneAnno
Zachary Stoebner, David Lu, Seok Hee Hong, Nicholas Kavoussi, Ipek Oguz

SPIE Medical Imaging: Image Processing 2022

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

### Scripts
1. Run train.py, test.py, and synthesize.py from the slurm directory like so: 
```
python ../<phase>.py [--<argument name> <arg value> ...]
```
*If running from root, you should uncomment any `os.chdir('..')` instructions in main script flow; the default behavior is to call from slurm dir for running on a cluster.*

2. From the project root, input the command:
```
python scripts/<script>.py [--<argument name> <arg value> ...]
```


## Troubleshooting
- The model is highly sensitive to normalization, and tends to perform worse when the training images are normalized. However, **if the model trains on normalized inputs, then inputs for testing and synthesis must also be normalized for the model.** 


