import click
from pathlib import Path
from joblib import dump
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .pipeline import create_pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_validate


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--scaling",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--select_feature",
    default=None,
    type=click.Choice([None, "select_from_model", "pca"]),
    show_default=True,
)
@click.option(
    "--which_model",
    default="random_forest",
    type=click.Choice(["random_forest", "log_regr"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=click.Choice(["gini", "entropy"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--max_depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--penalty",
    default="l2",
    type=click.Choice(["l1", "l2", "none"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--solver",
    default="lbfgs",
    type=click.Choice(["lbfgs", "newton-cg", "sag", "saga"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--fit_intercept",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max_iter",
    default=10000,
    type=int,
    show_default=True,
)
@click.option(
    "--random_state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--cv_k_split",
    default=5,
    type=int,
    show_default=True,
)
def train(dataset_path: Path, save_model_path: Path, scaling: bool, select_feature, which_model,
          n_estimators, criterion, max_depth,
          penalty, solver, c, fit_intercept, max_iter,
          random_state, cv_k_split):

    df = pd.read_csv(dataset_path)
    X = df.drop(columns='Cover_Type')
    y = df['Cover_Type']

    with mlflow.start_run():
        model = create_pipeline(scaling, select_feature, which_model,
                                n_estimators, criterion, max_depth,
                                penalty, solver, c, fit_intercept, max_iter, random_state)

        cv_inner = KFold(n_splits=cv_k_split, random_state=1, shuffle=True)
        space = dict()
        if which_model == 'random_forest':
            space['mod__n_estimators'] = [50, 100, 200, 300]
            space['mod__criterion'] = ['gini', 'entropy']
            space['mod__max_depth'] = [100, 200, 500, 1000]
        elif which_model == 'log_regr':
            space['mod__C'] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
            space['mod__fit_intercept'] = [False, True]

        search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)

        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        scores = cross_validate(search, X, y,
                                scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted'], cv=cv_outer,
                                n_jobs=-1)
        

        mlflow.log_param("model", which_model)
        mlflow.log_param("feature_selector", select_feature)
        if which_model == "random_forest":
            mlflow.log_param("use_scaler", scaling)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)
        elif which_model == "log_regr":
            mlflow.log_param("use_scaler", True)
            mlflow.log_param("penalty", penalty)
            mlflow.log_param("c", c)
            mlflow.log_param("fit_intercept", fit_intercept)
            mlflow.log_param("max_iter", max_iter)

        mlflow.log_metric("accuracy", np.mean(scores["test_accuracy"]))
        mlflow.log_metric("F1", np.mean(scores["test_f1_weighted"]))
        mlflow.log_metric("ROC AUC", np.mean(scores["test_roc_auc_ovr_weighted"]))
        mlflow.sklearn.log_model(model, which_model)

        click.echo(
            'Accuracy mean: %.3f, with std: %.3f' % (np.mean(scores["test_accuracy"]), np.std(scores["test_accuracy"])))
        click.echo(
            'F1 mean: %.3f, with std: %.3f' % (np.mean(scores["test_f1_weighted"]), np.std(scores["test_f1_weighted"])))
        click.echo('ROC AUC mean: %.3f, with std: %.3f' % (
            np.mean(scores["test_roc_auc_ovr_weighted"]), np.std(scores["test_roc_auc_ovr_weighted"])))

        result = search.fit(X, y)
        best_model = result.best_estimator_
        click.echo('Best parameters')
        click.echo(search.best_params_)
        y_pred = best_model.predict(X)

        dump(best_model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
