from click.testing import CliRunner
import pytest
import os
from random import randint
import numpy as np
import pandas as pd

from faker import Faker
from pathlib import Path

from forest_cover_type.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def generate_random_dataset(path):
    rand_data_row = np.random.randint([[1], [1], [10], [10], [50], [50], [100], [100], [200], [200]],
                                      [[10], [10], [50], [50], [100], [100], [200], [200], [500], [500]], size=(10, 5))
    target_row = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    rand_data = np.c_[rand_data_row, target_row]
    df = pd.DataFrame(rand_data, columns=['1_feat', '2_feat', '3_feat', '4_feat', '5_feat', 'Cover_Type'])
    df.to_csv(path)


def test_correctness_without_auto_grid_search(runner: CliRunner):
    """It checks the correctness of train function with auto_grid_search = False"""
    with runner.isolated_filesystem():
        generate_random_dataset('random_data.csv')
        result = runner.invoke(
            train,
            ["-d", 'random_data.csv', "-s", 'model.joblib', "--auto_grid_search", False, "--select_feature", 'pca'],
        )
        assert result.exit_code == 0


def test_correctness_for_log_regr(runner: CliRunner):
    """It checks the correctness of train function with scaling = False"""
    with runner.isolated_filesystem():
        generate_random_dataset('random_data.csv')
        result = runner.invoke(
            train, ["-d", 'random_data.csv', "-s", 'model.joblib', "--which_model", 'log_regr'],
        )
        assert result.exit_code == 0


def test_error_for_invalid_select_feature(runner: CliRunner):
    """It fails when select_feature has incorrect value"""
    with runner.isolated_filesystem():
        generate_random_dataset('random_data.csv')
        result = runner.invoke(
            train, ["-d", 'random_data.csv', "-s", 'model.joblib', "--select_feature", 2],
        )
        assert result.exit_code == 2
        assert "Invalid value for '--select_feature'" in result.output


def test_error_for_invalid_which_model(runner: CliRunner):
    """It fails when which_model has invalid value"""
    fake = Faker()
    with runner.isolated_filesystem():
        generate_random_dataset('random_data.csv')
        result = runner.invoke(
            train, ["-d", 'random_data.csv', "-s", 'model.joblib', "--which_model", fake.word()],
        )
        assert result.exit_code == 2
        assert "Invalid value for '--which_model'" in result.output
