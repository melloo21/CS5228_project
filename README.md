# CS5228_project
The goal of this task is to predict the resale price of a car based on its properties (e.g., make, model, mileage, age, power, etc).

Kaggle link: https://www.kaggle.com/competitions/cs-5228-2410-final-project

To reproduce envrionment: 

```{bash}
conda env create -f environment.yml
```

## Code Run Process

1. Run dataset_preprocessing.py to obtain dataset. This script performs the relevant data imputations, handling of outliers, feature transformations and saves the relevant train, validation and test datasets.

```{bash}
python dataset_preprocessing.py --method SimpleImputers_OutliersRemoved_NoVehCond
```
2. Feature Engineering/Selection and Model Selection
- feature_model_selection.ipynb: Includes the visualisation plots of feature transformation, feature selection methods and the learning curves to select the model

3. Model training
- We use GuildAI to track and monitor each run's progress. For each model that were deemed promising using the learning curve approach, we run GridSearch to identify the best set of hyperparameters.

For XGB grid search:

```{bash}
guild run train_xgb_randsearch.py
```

For LightGBM grid search:

```{bash}
guild run train_lgbm.py
```

For Gradient Boosted Trees grid search:

```{bash}
guild run train_gbtrees.py
```

4. Model Explanability:

- Model Explanability.ipynb: Includes the scripts used for final model training, code submission and model explanability using SHAP and LIME methods. 


## Folders
- model_assets: collection of best model weights
- processed_dataset: collection of preprocessed csv that is obtained via running data_processing.py
- superceeded_notebooks: rough working, function testing and sanity check
- utils: utility functions required for data preprocessing, modelling and model evaluation