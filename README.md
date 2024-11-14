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

2. Model training
- We use GuildAI to track and monitor each run's progress. For each model that are deemed promising, we run GridSearch to identify the best set of hyperparameters.

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

3. Model Explanability:

- Model Explanability.ipynb: Includes the scripts used for final model training, code submission and model explanability using SHAP and LIME methods. 