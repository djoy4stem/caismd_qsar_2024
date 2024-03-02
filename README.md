
## About
caismd_qsar_2024 provides material for the [CAiSMD 2024](https://bharatyuva.org/index.php/caismd) Hands-on Session: Exploring Antimalarial Drug Discovery Through Cheminformatics and QSAR modeling. \
An introduction to the tutorial is provided in the .pdf document.

## Installation
The tutorial is provided through a series of notebooks in the data/ folder. It is recommended to install an IDE that can incorportate Jupyter notebook extensions, such vas [VSCode](https://code.visualstudio.com/).

1. Create and activate a new conda environment (Preferrably with Python version 3.9)
    * Run ```conda create -n caismd_qsar_2024 python=3.9```. 
    * Run ```conda activate caismd_qsar_2024``` .
  
2. Install packages with pip
    * Run ```pip install mordred rdkit-pypi pandas umap-learn matplotlib seaborn```
    * Run ```pip install scikit-learn optuna lightgbm shap```
    * Run ```pip install ipywidgets ipykernel```

3. In order to view and interact with the Jupyer notebook, it is important to regiuster the kernel that run the created conda environment
    * Run ```python -m ipykernel install --user --name=caismd_qsar_2024```

## Prerequisites
The following prerequisites are recommended to the reader. However, the basic understanding of chemistry is not a must.

* Basic understanding of chemistry concepts.
* Basic understanding of statistical analysis or machine learning concepts.
* Familiarity with molecular structures and chemical descriptors.
* Comfortable working with data.
* Some experience with (Python) programming or a willingness to learn basic programming concepts
