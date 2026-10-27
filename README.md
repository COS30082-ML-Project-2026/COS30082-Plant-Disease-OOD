## Compositional prototype retrieval (approach 4)

DINOv2 ViT-B/14 with learned crop/disease projections, partner-balanced training
prototypes, zero-shot pair composition and supporting-image retrieval.

- [Six-section notebook](approaches/approach4_compositional_prototype_retrieval.ipynb)
- [Scope, methodology and task distribution](docs/approach4_scope.md)
- [Colab setup and training](colab/README.md)

Default is a DEBUG smoke test. For the full benchmark, prepare DPD-Small and run
`python -m colab.run_approach4 --full`. Full and DEBUG outputs are separated.
Tests: `python -m unittest discover -s tests -v`.

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



## Python Run
Run once after training: 
```cmd
python build_index.py

```

Run app.py
```cmd
streamlit run app.py

```
