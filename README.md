Final project for RS School Machine Learning course.

This project uses [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Usage
This package allows you to train model for predicting the forest cover type (the predominant kind of tree cover) from strictly cartographic variables.
1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.0.0).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command (you will run the code with GridSearch function):
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) and experiment by your own by changing --auto_grid_search = False parameter. 
```sh
poetry run train --auto_grid_search=False
```
To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
7. To see the pandas profiling information, please run eda code:
```sh
poetry run eda
```
You will find the report on this path src/forest_cover_type/EDA.

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [tox](https://tox.wiki/en/latest/): 
```
tox
```
Format your code with [black](https://github.com/psf/black):
```
poetry run black src
```

## Results in the MLFlow UI:

![Screenshot](src/forest_cover_type/Screenshots/MLFlow_screenshot.JPG?raw=true "Title")

## All tests with PyTest passed:

![Screenshot](src/forest_cover_type/Screenshots/Passed_tests.JPG?raw=true "Title")

## Formatted with black:

![Screenshot](src/forest_cover_type/Screenshots/black.JPG?raw=true "Title")

## Flake8 passed:

![Screenshot](src/forest_cover_type/Screenshots/Formatted_flake8.JPG?raw=true "Title")

## Mypy report:

![Screenshot](src/forest_cover_type/Screenshots/mypy.JPG?raw=true "Title")

## Tox report:

![Screenshot](src/forest_cover_type/Screenshots/tox_report.JPG?raw=true "Title")