from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline


def create_pipeline(scaling: bool, select_feature: bool, n_estimators: int, criterion, max_depth, random_state: int):
    model = RandomForestClassifier(n_estimators, criterion, max_depth, random_state, n_jobs=-1)
    scaler = None
    select = None
    if scaling:
        scaler = StandardScaler()

    if select_feature:
        select = SelectFromModel(model)

    return Pipeline([
        ('sca', scaler),
        ('sel', select),
        ('mod', model)
    ])
