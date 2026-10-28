## Create Conda Environment
```cmd
conda create -n cos30082mlp python=3.11 -y
conda activate cos30082mlp

```

## Update Python Package
```cmd
# Upgrade
python -m pip install --upgrade pip

```

## Install Requirements (CUDA Version)
```
pip install -r requirements.txt

```

## Run Application
Download Dataset:
```cmd
python download_data.py

```
---
Run once after training: 
```cmd
python build_index.py

```
---
Run app.py
```cmd
streamlit run app.py

```