ip5wke
==============================

Image Recognition of Machine Parts, based on VGGNet 16 using Tensorflow

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── Android           <- Android App
    │   │   ├── classimages           <- Contains display image for each class
    │   │   ├── android.txt    <- Contains name and author of the App
    │   │   ├── buildozer.spec <- Contains buildozer build configuration 
    │   │   ├── main.py        <- The main Application
    │   │   └── takepicture.kv <- App UI
    │   │
    │   ├── data           <- Scripts to generate data
    │   │   ├── make_dataset.py            <- Creates the dataset from raw images
    │   │   ├── create_train_valid_test.py <- Splits the dataset into train, validation and testing sets (70%, 15%, 15%)
    │   │   ├── create_rotated_images.py   <- Rotates images to create synthetic dataset
    │   │   └── resize_and_crop.py         <- methods for resizing and cropping images
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── export_serving_model.py   <- simplifies a model for production use
    │   │   ├── ip5wke.py                 <- contains model definition
    │   │   ├── ip5wke_input.py           <- reads images for training/evaluation
    │   │   ├── ip5wke_multi_gpu.py       <- For multi-gpu training (untested)
    │   │   ├── model_pb2.py              <- needed for google RPC to work
    │   │   ├── predict_pb2.py            <- needed for google RPC to work
    │   │   ├── prediction_service_pb2.py <- needed for google RPC to work
    │   │   ├── rest_api.py               <- Contains the REST API for the mobile App
    │   │   ├── test_model.py             <- tests and evaluates a model on validation or test set
    │   │   └── train_model.py            <- trains a model
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   ├── hyperparam_search.py             <- visualizes hyper parameter search results
    │       └── next_parameter.py                     <- gets next random hyperparameters to try
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
