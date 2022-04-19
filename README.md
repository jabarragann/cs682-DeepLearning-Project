# CS682 final project: Unsupervised classification of surgical gestures 

## Setup instructions
Clone the repo with the recursive flag included.
```
git clone <repo-url> --recursive
```

Then, make sure to update the data's paths in `deepgesture/.project_paths`.

### Option 1:
Create a new conda environment with the custom modules added
```
conda env create -f environment.yml
```

### Option 2:
Add custom modules via pip

```
pip install -e .
pip install -e ./torch-suite/
pip install -e ./torch-suite/pytorch-checkpoint/
```

# Other useful commands

```
pip3 install torch==1.11.0 torchvision torchaudio
```