from ast import Assert
import json
import joblib
import numpy as np
# import lightgbm as lgb
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go     
from plotly.subplots import make_subplots
from IPython.display import IFrame, display_html

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score


class RegressionEvaluate:

    def __init__(self, model_weight_path):

        self.model_weight_path = model_weight_path

    def _load_model(self, model_name: str):

        return joblib.load(f"{self.model_weight_path}/{model_name}.joblib")

    def _get_predicted_r2(
        self, x_values: np.ndarray, y_values: pd.Series, y_predict: np.ndarray
    ):

        X = x_values

        hat = X.dot(np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)))

        denom = 1 - np.diag(hat)

        resid = np.abs(y_predict - y_values)

        PRESS = np.sum((resid / denom) ** 2)

        SST = np.sum((y_values - y_values.mean()) ** 2)

        pred_r2 = 1 - PRESS / SST

        return pred_r2

    def _get_adjusted_r2(self, x, y, y_pred) -> str:

        return str(
            1 - ((1 - r2_score(y, y_pred)) * (len(x) - 1) / (len(x) - len(x[0]) - 1))
        )

    def _infer_type(self, data) -> np.ndarray:

        ## type checker

        try:

            assert type(data) == np.ndarray, "incorrect type"

        except AssertionError:

            data = data.to_numpy()

            print("INFO [_infer_type] :: Type changed")

            return data

        else:

            return data

    def regression_evaluate(
        self, df_train_tuple: tuple, df_valid_tuple: tuple, model_name: tuple
    ):

        model = self._load_model(model_name)
        X_train, y_train = df_train_tuple[0], df_train_tuple[1]
        X_valid, y_valid = df_valid_tuple[0], df_valid_tuple[1]

        ## type checker

        X_train = self._infer_type(X_train)
        y_train = self._infer_type(y_train)
        X_valid = self._infer_type(X_valid)
        y_valid = self._infer_type(y_valid)

        try:

            # Train and validation scores

            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)

            train_message = {
                "Train RMSE ::": str(
                    np.sqrt(mean_squared_error(y_train, y_train_pred))
                )[:6],
                "Train MAE": str(np.mean(abs(y_train - y_train_pred)))[:6],
                "Train PMAE": str(np.mean(abs(y_train - y_train_pred) / y_train) * 100)[
                    :6
                ],
                "Train PME ": str(np.mean(1 - y_train_pred / y_train) * 100)[:6],
                "Train R2": str(r2_score(y_train, y_train_pred))[:5],
                "Train Adj R2": self._get_adjusted_r2(X_train, y_train, y_train_pred),
                "Train Predicted R2": str(
                    self._get_predicted_r2(X_train, y_train, y_train_pred)
                ),
            }

            valid_message = {
                "Valid RMSE ::": str(
                    np.sqrt(mean_squared_error(y_valid, y_valid_pred))
                )[:6],
                "Valid MAE": str(np.mean(abs(y_valid - y_valid_pred)))[:6],
                "Valid PMAE": str(np.mean(abs(y_valid - y_valid_pred) / y_valid) * 100)[
                    :6
                ],
                "Valid PME ": str(np.mean(1 - y_valid_pred / y_valid) * 100)[:6],
                "Valid R2": str(r2_score(y_valid, y_valid_pred))[:5],
                "Valid Adj R2": self._get_adjusted_r2(X_valid, y_valid, y_valid_pred),
                "Valid Predicted R2": str(
                    self._get_predicted_r2(X_valid, y_valid, y_valid_pred)
                ),
            }

            print(train_message, valid_message)


        except Exception as e:

            print(f"ERROR [regression_evaluate] :: {e}")

        # Visualise scatter plot for validation data

        data_plot = pd.DataFrame({"Predicted": y_valid_pred, "Actual": y_valid})
        fig_scatter = px.scatter(data_plot, x="Actual", y="Predicted")
        fig_line = px.line(data_plot, x="Actual", y="Actual")

        fig_line.update_traces(line_color="#d62728", line_width=3)

        fig = go.Figure(data=fig_scatter.data + fig_line.data)

        fig.update_layout(height=550, title="y_valid predicted vs actual")

        fig.update_xaxes(
            title_text="Actual Values",
            dtick=10,
            showline=True,
            linewidth=2,
            gridwidth=2,
            linecolor="grey",
            mirror=True,
        )

        fig.update_yaxes(
            title_text="Predicted Values",
            dtick=10,
            showline=True,
            linewidth=2,
            showgrid=False,
            gridwidth=2,
            linecolor="grey",
            mirror=True,
        )

        fig.show(renderer="svg")

        data_plot = pd.DataFrame({"Predicted": y_train_pred, "Actual": y_train})

        fig_scatter = px.scatter(data_plot, x="Actual", y="Predicted")

        fig_line = px.line(data_plot, x="Actual", y="Actual")

        fig_line.update_traces(line_color="#d62728", line_width=3)

        fig = go.Figure(data=fig_scatter.data + fig_line.data)

        fig.update_layout(height=550, title="y_train predicted vs actual")

        fig.update_xaxes(
            title_text="Actual Values",
            dtick=10,
            showline=True,
            linewidth=2,
            gridwidth=2,
            linecolor="grey",
            mirror=True,
        )

        fig.update_yaxes(
            title_text="Predicted Values",
            dtick=10,
            showline=True,
            linewidth=2,
            showgrid=False,
            gridwidth=2,
            linecolor="grey",
            mirror=True,
        )

        fig.show(renderer="svg")


        ## Plotting residuals

        valid_res_scatter = px.scatter(x=y_valid_pred, y=(y_valid_pred - y_valid))

        train_res_scatter = px.scatter(x=y_train_pred, y=(y_train_pred - y_train))

        fig = go.Figure(data=valid_res_scatter.data + train_res_scatter.data)

        fig.update_layout(height=550, title="Residual Plot")

        fig.update_xaxes(
            title_text="Index",
            dtick=10,
            showline=True,
            linewidth=2,
            gridwidth=2,
            linecolor="grey",
            mirror=True,
        )

        fig.update_yaxes(
            title_text="Residual Values",
            dtick=10,
            showline=True,
            linewidth=2,
            showgrid=False,
            gridwidth=2,
            linecolor="grey",
            mirror=True,
        )

        fig.show(renderer="svg")

    def mini_reg_evaluate(
        self, df_train_tuple: tuple, df_valid_tuple: tuple, model_name: str, save_path:str
    ):

        model = self._load_model(model_name)
        X_train, y_train = df_train_tuple[0], df_train_tuple[1]
        X_valid, y_valid = df_valid_tuple[0], df_valid_tuple[1]

        ## type checker

        X_train = self._infer_type(X_train)
        y_train = self._infer_type(y_train)
        X_valid = self._infer_type(X_valid)
        y_valid = self._infer_type(y_valid)

        # Train and validation scores
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        
        try:

            # Train and validation scores

            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)

            train_message = {
                "Train RMSE ::": str(
                    np.sqrt(mean_squared_error(y_train, y_train_pred))
                )[:6],
                "Train MAE": str(np.mean(abs(y_train - y_train_pred)))[:6],
                "Train PMAE": str(np.mean(abs(y_train - y_train_pred) / y_train) * 100)[
                    :6
                ],
                "Train PME ": str(np.mean(1 - y_train_pred / y_train) * 100)[:6],
                "Train R2": str(r2_score(y_train, y_train_pred))[:5],
                "Train Adj R2": self._get_adjusted_r2(X_train, y_train, y_train_pred),
                "Train Predicted R2": str(
                    self._get_predicted_r2(X_train, y_train, y_train_pred)
                ),
            }

            valid_message = {
                "Valid RMSE ::": str(
                    np.sqrt(mean_squared_error(y_valid, y_valid_pred))
                )[:6],
                "Valid MAE": str(np.mean(abs(y_valid - y_valid_pred)))[:6],
                "Valid PMAE": str(np.mean(abs(y_valid - y_valid_pred) / y_valid) * 100)[
                    :6
                ],
                "Valid PME ": str(np.mean(1 - y_valid_pred / y_valid) * 100)[:6],
                "Valid R2": str(r2_score(y_valid, y_valid_pred))[:5],
                "Valid Adj R2": self._get_adjusted_r2(X_valid, y_valid, y_valid_pred),
                "Valid Predicted R2": str(
                    self._get_predicted_r2(X_valid, y_valid, y_valid_pred)
                ),
            }

            print(train_message, valid_message)


        except Exception as e:

            print(f"ERROR [regression_evaluate] :: {e}")

        # Visualize scatter plot for validation data
        plt.scatter(y_valid, y_valid_pred, color='blue', alpha=0.5)
        plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color='red', linestyle='--')
        plt.title("y_valid predicted vs actual")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
        
        # Visualize scatter plot for training data
        plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
        plt.title("y_train predicted vs actual")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
        
        # Visualize residual plot
        plt.scatter(y_valid_pred, y_valid_pred - y_valid, color='blue', alpha=0.5)
        plt.plot([y_valid_pred.min(), y_valid_pred.max()], [0, 0], color='red', linestyle='--')
        plt.title("Residual Plot (Validation Data)")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residual Values")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
        
        plt.scatter(y_train_pred, y_train_pred - y_train, color='blue', alpha=0.5)
        plt.plot([y_train_pred.min(), y_train_pred.max()], [0, 0], color='red', linestyle='--')
        plt.title("Residual Plot (Training Data)")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residual Values")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
    
    # feature importance

    def _plot_feature_immportance(
        self,
        model_name: str,
        feature_importance: list,
        asset_path: str = "/home/jupyter/pdm-facilities-training/assets/model_details",
    ) -> dict:

        try:

            cols = json.load(open(f"{asset_path}/{model_name}_col.json"))

            fig = px.bar(x=feature_importance, y=cols["col_order"], orientation="h")

            fig.update_layout(
                barmode="stack", yaxis={"categoryorder": "total descending"}
            )

        except Exception as e:

            print(f"[_plot_feature_immportance] :: {e}")

        return fig

    def _plot_perm_feature_immportance(
        self,
        model_name: str,
        data: tuple,
        n_repeats: int = 10,
        random_state: int = 0,
        asset_path: str = "/home/jupyter/pdm-facilities-training/assets/model_details",
    ):

        # recreating the model

        best_model = self._load_model(model_name)

        x, y = data[0], data[1]

        result = permutation_importance(
            best_model.fit(x, y), x, y, n_repeats=n_repeats, random_state=random_state
        )

        # printing total number of scores

        print("Permutation importance scores", result.importances)

        # plotting the fig immportance and std dev

        cols = json.load(open(f"{asset_path}/{model_name}_col.json"))

        perm_sorted_idx = result.importances_mean.argsort()

        cols = pd.Series(cols["col_order"]).reindex(index=perm_sorted_idx)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        ax1.barh(cols, result.importances_mean[perm_sorted_idx])

        ax2.barh(cols, result.importances_std[perm_sorted_idx])

        ax1.set_xlabel("Importance Mean")

        ax2.set_xlabel("Importance std")

        fig.suptitle("Permutation Importance")

        plt.show()

    def hierarchy_plots(
        self,
        model_name: str,
        data: tuple,
        asset_path: str = "/home/jupyter/pdm-facilities-training/assets/model_details",
    ):

        cols = json.load(open(f"{asset_path}/{model_name}_col.json"))

        cols = cols["col_order"]

        X = data[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        corr = spearmanr(X).correlation

        # Ensure the correlation matrix is symmetric

        corr = (corr + corr.T) / 2

        np.fill_diagonal(corr, 1)

        # We convert the correlation matrix to a distance matrix before performing

        # hierarchical clustering using Ward's linkage.

        distance_matrix = 1 - np.abs(corr)

        dist_linkage = hierarchy.ward(squareform(distance_matrix))

        dendro = hierarchy.dendrogram(
            dist_linkage, labels=cols, ax=ax1, leaf_rotation=90
        )

        dendro_idx = np.arange(0, len(dendro["ivl"]))

        ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])

        ax2.set_xticks(dendro_idx)

        ax2.set_yticks(dendro_idx)

        ax2.set_xticklabels(dendro["ivl"], rotation="vertical")

        ax2.set_yticklabels(dendro["ivl"])

        fig.tight_layout()

        plt.show()
