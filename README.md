# CGAN-SAE

This repository is the official implementation of CGAN-SAE.

## Requirements

To install requirements:

`pip install -r requirements.txt`

## Data

The `data` folder contains all raw data files (in `.txt` format) and the corresponding cleaned datasets (in `.csv` format). The model requires the cleaned `.csv` files as input.

The original raw datasets are publicly available in the GEO database under the following accession numbers:

- **GSE39635**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39635
- **GSE47460**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE47460

The data preprocessing and cleaning procedures follow those described in the original publication.

## src

All model implementations, including the proposed method and baseline models, are located in the `src` directory. The default dataset is the **rice dataset**; to switch to another dataset or use your own, please modify the data path accordingly in the script.

The code can be executed in **PyCharm** after installing the required dependencies. Ensure that:

- The correct Python interpreter (with all dependencies installed) is selected in PyCharm.
- The dataset path in the code points to your local `.csv` files.

Once configured, you can run the model directly by clicking the "Run" button.