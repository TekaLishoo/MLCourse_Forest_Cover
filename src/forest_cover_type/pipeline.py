from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier


def create_pipeline(scaling: bool, select_feature: bool, model, n_estimators: int, criterion, max_depth,
                    alpha: float, fit_intercept: bool, norm: bool, solver,
                    random_state: int, ):
    if model == 'random_forest':
        model_cl = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          random_state=random_state, n_jobs=-1)
    elif model == 'ridge':
        model_cl = RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, normalize=norm, solver=solver,
                                   random_state=random_state)

    scaler = None
    select = None
    if scaling:
        scaler = StandardScaler()

    if select_feature:
        select = SelectFromModel(model)

    return Pipeline([
        ('sca', scaler),
        ('sel', select),
        ('mod', model_cl)
    ])
