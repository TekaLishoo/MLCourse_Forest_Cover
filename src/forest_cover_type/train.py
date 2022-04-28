import click
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def train(dataset_path: Path):
    df = pd.read_csv(dataset_path)
    X_train, X_val, y_train, y_val = train_test_split(df.drop(columns='Cover_Type'), df['Cover_Type'], test_size=0.2,
                                                      stratify=df['Cover_Type'], random_state=1)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    acc_train = accuracy_score(y_train, rf.predict(X_train))
    acc_val = accuracy_score(y_val, y_pred)
    print(f'Train accuracy: {acc_train}')
    print(f'Val accuracy: {acc_val}')
