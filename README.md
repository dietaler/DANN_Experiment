# DANN_Experiment
## Download the datasets
Please download the dataset from [here](https://speed.cs.nycu.edu.tw:28000/index.php/s/2j2PD3RK27JMH6s).

## Intrusion-Detection-System-Using-CNN-and-Transfer-Learning
This project generates tabular or image datasets to train our models.
### Description:
- `pre-processing_cicids2017_to6class.ipynb`: Converts `all_data.csv` to `all_data_6class.csv`, which has only 6 major classes.
- `pre-processing_cicids2017_RGB.ipynb`: Converts `all_data.csv` file to (1,224,224) images dataset, stored in folder `image_gray`.
- `pre-processing_cicids2017_gray.ipynb`: Converts `all_data.csv` file to (3,224,224) images dataset, stored in folder `image_RGB`.
- `pre-processing_cicids2018_to6class.ipynb`: Converts `all_data.csv` to `all_data_6class.csv`, which has only 6 major classes.
- `pre-processing_cicids2018_RGB.ipynb`: Converts `all_data.csv` file to (1,224,224) images dataset, stored in folder `image_gray`.
- `pre-processing_cicids2018_gray.ipynb`: Converts `all_data.csv` file to (3,224,224) images dataset, stored in folder `image_RGB`.

## DANN_image
This project trains DANN models on image datasets.

### How to run
```
cd DANN_image
python main.py
```

### Description:
- Modify the `main.py` and `train.py` when you want to switch the source and target datasets.
- Hyperparameters are set in `params.py`

## DANN_tabular
This project trains DANN models on tabular datasets.

### How to run
```
cd DANN_image
python main.py \
--source_data data/tabular_6class/cicids2017/all_data_6class.csv \
--target_data data/tabular_6class/cicids2018/all_data_6class.csv

or

python main.py \
--source_data data/tabular_6class/cicids2018/all_data_6class.csv \
--target_data data/tabular_6class/cicids2017/all_data_6class.csv
```

### Description:
- You can specify the source and target datasets when running the command.
- Hyperparameters are set in `params.py`