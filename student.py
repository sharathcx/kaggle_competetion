import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

train = pd.read_csv(r'C:\Users\shara\OneDrive\Desktop\data-science\data\playground-series-s4e6\train.csv')
test = pd.read_csv(r'C:\Users\shara\OneDrive\Desktop\data-science\data\playground-series-s4e6\test.csv')

model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 200],
            'criterion': ["gini", "entropy", "log_loss"],
            "min_samples_split": [x for x in range(10)],
            "bootstrap": [True, False]
        }
    },
    "xgb": {
        'model': GradientBoostingClassifier(),
        'params': {
            "loss": ["log_loss", "exponential"],
            "n_estimators": [50, 100, 150, 200],
            "criterion": ["friedman_mse", "squared_error"]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(train.drop('Target', axis=1), train['Target'])
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(df)
