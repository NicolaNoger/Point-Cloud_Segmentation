# PA-1 
**Authors**: Klaver Florian, Noger Nicola

## Point Cloud Segmentation with PointNet & PointNet++

**semantic segmentation of 3‑D point‑clouds** using two neural architectures:

| Directory         | Model                                                                     | DL Framework       |
| ----------------- | ------------------------------------------------------------------------- | ------------------ |
| `code/PointNet`   | **PointNet** (Qi *et al.*, 2017)                                          | TensorFlow / Keras |
| `code/PointnetPP` | **PointNet++** (Qi *et al.*, 2017) – fork of Erik Wijmans’ implementation "PointNet2_PyTorch" | PyTorch            |

- Both Models are completely independant from eachother. From the data loading to the training and testing.



## 1. Repository Road‑Map

<details>
<summary>Expand for a full tree</summary>

```text
PA-1/
├── code/
│   ├── PointNet/                  # 
|   |   ├── environment.yml        #
│   │   ├── label_map.json         #   
│   │   ├── pointnet_segmentation.keras  #   
│   │   ├── pointnet.ipynb         #   
│   │   ├── test_files_preparation.ipynb #   
│   │   ├── scaler.pkl             #   
│   │   └── README_PointNet.md     #  
│   │
│   └── PointnetPP/                # 
│       ├── PointNet2_PyTorch/     #   
│       │   ├── pointnet2          #    
│       │   ├── pointnet2_ops_lib  #     
│       │   └── pointnet2.egg-info #    
│       │
│       ├── Dataloader.ipynb       #   
│       ├── data_exploration.ipynb #   
│       ├── pointnet_spg_fps2.py   #   
│       ├── pointnetpp.sh          #  
│       ├── confusion_matrix.png   #  
│       └── requirements.txt       #   
│
├── docs/                          # 
├── datasets/                      # 
├── LICENSE
└── README.md (← you are here)
```

</details>





## 2. Installation 
### 2.1 Clone Repository

```bash
# clone repo and enter
$ git clone https://github.zhaw.ch/nogernic/PA-1.git && cd PA-1


```

### 2.2 PointNet

```bash
#create a ennvironment with requironments
conda env create -f environment.yml
```


### 2.3 PointNet++

```bash
# create a environment 
$ python -m venv .venv && source .venv/bin/activate

# install requironments
$ pip install -r code/PointnetPP/requirements.txt


```




## 3. Usage

### 3.1 PointNet (Keras)
This model is designed to run localy on CPU or GPU.

The file `pointnet.ipynb` contains the data preprocessing and the full PointNet model. In addition there is an extended model testing on the different municipalities we chose to explore. 

The notebook `test_files_preparation.ipynb` crops the data to the desired regions of the cities, so it is ready for the analysis.

### 3.2 PointNet++ (PyTorch)
This model is configured to run on the ZHAW earth-cluster on GPU. The setup can also be done locally but also there should run on GPU. Additionally, the CUDA and CUDNN versions need be checked to make sure it works correctly. 

The notebook `Dataloader.ipynb` loads the LAZ data into train, val, test folder as .npy files. 

The PointNet++ model is found in the `pointnet_spg_fps2.py` file. There the model is trained also with the model testing, which creates a classification report and a confusion matrix as png. Model checkpoints are safed when the training is finished.

The `PointNet2_PyTorch` folder contains the full implementation of PointNet++ in PyTorch, so the model is ready to run. 







