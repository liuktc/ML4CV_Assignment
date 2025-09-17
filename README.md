<!-- Badges -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

# Open-World Semantic Segmentation (ML4CV Assignment)

This repository contains the implementation of an open-world semantic segmentation model using a pre-trained DINOv3 backbone and metric learning techniques. The model is designed to segment known classes while identifying unknown classes as anomalies.

## Setup

### Kaggle

To run the code on Kaggle, the requirements installation is handled in the `main.ipynb` notebook. Simply import the notebook to Kaggle and run the first cell. Make sure to add the train and test datasets to the Kaggle notebook environment.

- [Train dataset kaggle link](https://www.kaggle.com/datasets/lucadome/streethazards-train)
- [Test dataset kaggle link](https://www.kaggle.com/datasets/ginobellococco/streethazards-test)

### Locally

To run the code locally, create a virtual environment and install the required packages using the following command:

```bash
pip install -r requirements.txt
```

Make sure to download the train and test datasets and place them in the `data` folder. Your folder should look like this:

```
./data/
├── train
│   ├── images/
│   ├── annotations/
│   ├── train.odgt
│   └── validation.odgt
└── test
    ├── images/
    ├── annotations/
    └── test.odtg

```

- [Training and validation sets download link](https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar)
- [Test set download link](https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar)

## Training

To train a model use the file `train.py`. The training script allows you to choose between different metric learning losses (Triplet Loss, NT-Xent Loss, Proxy-Anchor Loss) or no metric learning loss at all. The model can be trained with or without a small CNN that extracts low-level features from the input image.
Example command to train the model with Triplet Loss and the small CNN:

```bash
python run.py --save_path "triplet_metric_fixed.pth" --model DinoMetricLearning --embedding_size 256 --loss TripletMargin --loss_weighting fixed --lambda_metric 1 --lambda_ce 1 --pixel_per_class 50 --kaggle --wandb
```

## Inference

The `main.ipynb` notebook contains the inference code for testing and visualization. Tu run it, simply import it to KAGGLE (and set the global constant at the top `KAGGLE = True`) or run it locally (set `KAGGLE = False`).

![Example Output](https://github.com/liuktc/ML4CV_Assignment/blob/62ef47c84dfc9df7f232196583aa8d1f25141a0c/output.png?raw=true)

## Contacts

For any questions or issues, please contact:

- Luca Domeniconi (luca.domeniconi5@studio.unibo.it)
