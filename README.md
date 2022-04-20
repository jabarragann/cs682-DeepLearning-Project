# CS682 final project: Unsupervised classification of surgical gestures 

# Setup instructions

## Step 1 (Clone repository):
Clone the repo with the recursive flag included.
```
git clone <repo-url> --recursive
```
If repository was clone without recursive flag execute.
```
git submodule update --init --recursive
```
Then, make sure to update the data's paths in `deepgesture/.project_paths`.

## Step 2 (Install packages)
**Option 1:**
Create a new conda environment with the custom modules added
```
conda env create -f environment.yml
```
**Option 2:**
Add custom modules via pip

```
pip install -r requirements.txt
pip install -e .
```
Install submodules
```
pip install -e ./torch-suite/
pip install -e ./torch-suite/pytorch-checkpoint/
```

# Other useful commands

```
pip3 install torch==1.11.0 torchvision torchaudio
```
Make sure you don't commit changes in the .project_paths files
```
git update-index --skip-worktree deepgesture/.project_paths 
```
