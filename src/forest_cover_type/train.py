import click
from pathlib import Path
from joblib import dump
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .pipeline import create_pipeline


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
def train(dataset_path: Path, save_model_path: Path, scaling: bool, select_feature: bool, n_estimators, criterion,
          max_depth, random_state):
    df = pd.read_csv(dataset_path)
    X_train, X_val, y_train, y_val = train_test_split(df.drop(columns='Cover_Type'), df['Cover_Type'], test_size=0.2,
                                                      stratify=df['Cover_Type'], random_state=1)
    model = create_pipeline(scaling, select_feature, n_estimators, criterion, max_depth, random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_val = accuracy_score(y_val, y_pred)
    print(f'Train accuracy: {acc_train}')
    print(f'Val accuracy: {acc_val}')

    f1_val = f1_score(y_val, y_pred, average='weighted')
    print(f'F1 score: {f1_val}')

    roc_auc_val = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')
    print(f'ROC AUC: {roc_auc_val}')

    dump(model, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
