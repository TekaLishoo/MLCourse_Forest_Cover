from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def create_pipeline(scaling: bool, select_feature, model, n_estimators: int, criterion, max_depth,
                    penalty, solver, c: float, fit_intercept: bool, max_iter,
                    random_state: int, ):
    if model == 'random_forest':
        model_cl = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          random_state=random_state, n_jobs=-1)
        scaler = None

    elif model == 'log_regr':
        model_cl = LogisticRegression(penalty=penalty, solver=solver, C=c, fit_intercept=fit_intercept,
                                      class_weight="balanced", max_iter=max_iter, n_jobs=-1, random_state=random_state)
        scaler = StandardScaler()

    select = None
    if scaling:
        scaler = StandardScaler()

    if select_feature == 'select_from_model':
        select = SelectFromModel(model_cl)

    if select_feature == 'pca':
        select = PCA(n_components=3)

    return Pipeline([
        ('sca', scaler),
        ('sel', select),
        ('mod', model_cl)
    ])
