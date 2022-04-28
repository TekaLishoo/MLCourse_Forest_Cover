from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline


def create_pipeline(scaling: bool, select_feature: bool):
    model = RandomForestClassifier()
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