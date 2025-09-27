# CGAN-SAE

This repository is the official implementation of CGAN-SAE.

## Environment Setup

To ensure the code runs correctly, we provide an `environment.yml` file to create a reproducible Conda environment. Please follow the steps below to set up the environment.

### 1. Install Conda

First, make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed. Miniconda is recommended.

### 2. Create the Conda Environment

After cloning this repository, run the following command in the project root directory:

`conda env create -f environment.yml`

This command will automatically create a Conda environment named cgan-sae_env and install all required dependencies as specified in the environment.yml file.

## Data

The `data` folder contains all raw data files (in `.txt` format) and the corresponding cleaned datasets (in `.csv` format). The model requires the cleaned `.csv` files as input.

The original raw datasets are publicly available in the GEO database under the following accession numbers:

- **GSE39635**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39635
- **GSE47460**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE47460

The data preprocessing and cleaning procedures follow those described in the original publication.

## src

All model implementations, including the proposed method and baseline models, are located in the `src` directory. The default dataset is the rice dataset; to switch to another dataset , please modify the data path accordingly in the script.

The code can be executed in PyCharm after installing the required dependencies. Ensure that:

- The correct Python interpreter (with all dependencies installed) is selected in PyCharm.
- The dataset path in the code points to your local `.csv` files.

Once configured, you can run the model directly by clicking the "Run" button.