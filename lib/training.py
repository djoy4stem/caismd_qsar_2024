


import numpy as np
import pandas as pd


import optuna

from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_with_optuna(model_params, num_trials, scoring_function, direction, train_val_test_data, target_column, standardize=False, n_jobs=1):
    def objective(trial):
        # Define parameters to optimize
        params = {}
        for param_name, param_value in model_params['params'].items():
            # print(param_name, param_value )
            # print(f"param_name={param_value} - {param_value.__class__} - {isinstance(param_value, bool)}" )
            if isinstance(param_value, (list, tuple)):
                if isinstance(param_value[0], bool):
                    # print([str(t) for t in param_value])
                    a = trial.suggest_categorical(param_name, ["True", "False"])
                    # print(f"a = {a}")
                    params[param_name] = bool(a)
                    # print(f"params[param_name] = {params[param_name]}")

                elif isinstance(param_value[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_value)
                    if params[param_name] in ["True", "False"]:
                        print(f"{params[param_name]} in ['True', 'False']")
                        params[param_name] = bool(params[param_name])
                        print(params[param_name].__class__)   

                elif isinstance(param_value[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_value[0], param_value[-1])

                elif isinstance(param_value[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_value[0], param_value[-1])

                else:
                    raise ValueError(f"Unsupported parameter type for param_value[0] ({param_value[0]}).")
            elif isinstance(param_value, bool):
                # print(f"param_value = {param_value}")
                params[param_name] = trial.suggest_categorical(param_name, [param_value])
            elif isinstance(param_value, int):
                params[param_name] = trial.suggest_int(param_name, param_value, param_value)
            elif isinstance(param_value, str):
                # print(param_value)
                if param_value in ["True", "False"]:
                    params[param_name] = trial.suggest_categorical(param_name, [bool(param_value)])
                else:
                    params[param_name] = trial.suggest_categorical(param_name, [param_value])
                # print(f"params[param_name] = {params[param_name]}")

            else:
                raise ValueError(f"Unsupported parameter type for param_value ({param_value}).")
        # Instantiate and fit the model
        scores = []
        for i in range(len(train_val_test_data)):
            fold = train_val_test_data[i]
            train_df = fold[0]
            val_df   = fold[1]

            if standardize:
                standardizer = StandardScaler()
                # print(f"fold[0]={fold[0].shape}\nfold[0][target_column].isna().sum() = {fold[0][target_column].isna().sum()}\n\n")
                train_df = pd.DataFrame(standardizer.fit_transform(fold[0].drop([target_column], axis=1)), columns=fold[0].columns.difference([target_column]))
                train_df.index = fold[0].index
                # print(target_column in train_df.columns)
                train_df[target_column] = fold[0][target_column]
                val_df   = pd.DataFrame(standardizer.fit_transform(fold[1].drop([target_column], axis=1)), columns=fold[1].columns.difference([target_column]))
                val_df.index = fold[1].index
                val_df[target_column] = fold[1][target_column]

                # print(train_df.shape, train_df.isna().sum())

            # print(f"{i} - Train {train_df.shape} - Val {val_df.shape}")

            model = model_params['model'](**params)


            if model_params['model'] in ['LGBMClassifier', 'LGBMRegressor']:
                model.fit(X=train_df.drop(columns=[target_column], axis=1)
                          , y=train_df[target_column]
                          , early_stopping = 50 ## to avoid overfitting
                        )

            else:
                model.fit(train_df.drop(columns=[target_column]), train_df[target_column])

            # Evaluate the model
            y_pred = model.predict(val_df.drop(columns=[target_column]))
            score = scoring_function(val_df[target_column], y_pred)
            # print(f"val_df[target_column] = {val_df[target_column]}")
            # print(roc_auc_score(y_true=val_df[target_column], y_score=y_pred))
            scores.append(score)
        # print(f"scores  = { scores }")
        mean_score = np.array(scores).mean()
        return np.array(mean_score)

    # Create a Optuna study
    study = optuna.create_study(direction=direction)

    # Optimize the study
    study.optimize(objective, n_trials=num_trials)

    # Get the best parameters
    best_params = study.best_params
    
    # Train the best model
    best_model = model_params['model'](**best_params)
    # print(train_val_test_data[0][:2])
    train_val_df = pd.concat(train_val_test_data[0][:2], axis=0)
    best_model.fit(train_val_df.drop(columns=[target_column]), train_val_df[target_column])

    return best_model, best_params, study.best_value, study.direction