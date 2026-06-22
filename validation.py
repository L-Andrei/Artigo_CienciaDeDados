from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Modelo base
rf = RandomForestClassifier(random_state=42)

# Valores que serão testados
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500]
}

# Grid Search
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Melhor n_estimators:", grid.best_params_['n_estimators'])
print("Melhor F1:", grid.best_score_)
