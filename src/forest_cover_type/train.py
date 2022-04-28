import click
from pathlib import Path
from joblib import dump
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
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
    default=False,
    type=bool,
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
    "--random_state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--cv_k_split",
    default=10,
    type=int,
    show_default=True,
)
def train(dataset_path: Path, save_model_path: Path, scaling: bool, select_feature: bool, n_estimators, criterion,
          max_depth, random_state, cv_k_split):
    df = pd.read_csv(dataset_path)
    X = df.drop(columns='Cover_Type')
    y = df['Cover_Type']

    model = create_pipeline(scaling, select_feature, n_estimators, criterion, max_depth, random_state)

    cv = KFold(n_splits=cv_k_split, random_state=1, shuffle=True)

    scores = cross_validate(model, X, y,
                            scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted'], cv=cv,
                            n_jobs=-1)
    print('Accuracy mean: %.3f, with std: %.3f' % (np.mean(scores["test_accuracy"]), np.std(scores["test_accuracy"])))
    print('F1 mean: %.3f, with std: %.3f' % (np.mean(scores["test_f1_weighted"]), np.std(scores["test_f1_weighted"])))
    print('ROC AUC mean: %.3f, with std: %.3f' % (
        np.mean(scores["test_roc_auc_ovr_weighted"]), np.std(scores["test_roc_auc_ovr_weighted"])))

    model.fit(X, y)
    y_pred = model.predict(X)

    dump(model, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
