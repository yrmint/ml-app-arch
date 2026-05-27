from collections import defaultdict
from copy import deepcopy

import optuna
import pandas as pd
import warnings


from sklearn.metrics import f1_score
from ml.core.kernel import SciKitCNN

warnings.filterwarnings('ignore')


class Objective(object):
    """
    Target function for hyperparams optimization
    Validation set is necessary for that case
    """

    def __init__(self, X, y, model_name, X_val, y_val, X_test, y_test):
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val.argmax(dim=1).numpy()
        self.X_test, self.y_test = X_test, y_test.argmax(dim=1).numpy()
        self.model_name = model_name
        # DF with results
        self.model_results_df = pd.DataFrame(
            columns=['n', 'Model', 'F1_val', 'F1_test', 'Parameters'])
        self.best_model = None
        self.best_model_est = None
        self.best_test_score = float('-inf')

    def __call__(self, trial):
        warnings.filterwarnings('ignore')
        # Defying model and its params bounds
        # Can be extended by adding elif clf_name == "***"
        clf_name = self.model_name

        if clf_name == "CNN":
            epoch_n = trial.suggest_int("epoch_n", 20, 80, 5)
            dropout = trial.suggest_float("dropout", 0.2, 0.5)
            batch_size = trial.suggest_int("batch_size", 32, 64)
            n_channels = trial.suggest_int("n_channels", 3, 5)
            first_channel = trial.suggest_int("first_channel", 16, 64, 16)
            channels = tuple(
                first_channel * (2 ** i) for i in range(0, n_channels)
            )
            clf_obj = SciKitCNN(
                lr=1e-4,
                batch_size=batch_size,
                channels=channels,
                dropout=dropout,
                epoch_n=epoch_n
            )
        else:
            raise ValueError(f"Unknown model: {clf_name}")

        print(f"\n{clf_name} hyperoptimization with params: "
              f"lr {1e-4}, channels {channels}, dropout {dropout}")

        # Model fitting and evaluating
        clf_obj.fit(self.X, self.y)
        y_val_pred = clf_obj.predict(self.X_val)
        y_test_pred = clf_obj.predict(self.X_test)
        print(self.y_val)
        print(y_val_pred)
        f1_val = f1_score(self.y_val, y_val_pred, average="macro")
        f1_test = f1_score(self.y_test, y_test_pred, average="macro")

        if f1_test > self.best_test_score:
            self.best_model = deepcopy(clf_obj)
            self.best_test_score = f1_test

        # Logging the DF
        self.model_results_df = pd.concat(
            [self.model_results_df, pd.DataFrame({
                'n': trial.number,
                'Model': clf_name,
                'Parameters': [trial.params],
                'F1_val': f1_val,
                'F1_test': f1_test
            })],
            ignore_index=True
        )

        return f1_val


class ModelOptimization:
    """
    Class for hyperparam search
    As the result - DF with logs
    """

    def __init__(self, models_list):
        self.models_list = models_list
        self.results_df = pd.DataFrame(
            columns=['n', 'Model', 'F1_val', 'F1_test', 'Parameters']
        )
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_f1_test = []
        self.studies = defaultdict(lambda: [])

    def fit(self, x, y,
            X_val, y_val,
            X_test, y_test,
            n_trials=20, n_startup_trials=10
            ):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        model = self.models_list[0]
        for s in ['tpe']:
            print(f"\n{model} hyperoptimization")

            if s == 'tpe':
                sampler = optuna.samplers.TPESampler(
                    multivariate=True,
                    n_startup_trials=n_startup_trials
                )
            elif s == 'cmaes':
                sampler = optuna.samplers.CmaEsSampler(
                    lr_adapt=True,
                    n_startup_trials=n_startup_trials
                )

            objective = Objective(
                x, y, model,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name="CNN_optimization",
                load_if_exists=True,
                storage="sqlite:///db.sqlite3"
            )
            study.set_metric_names(["F1_val"])
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True
            )
            self.best_models.append(objective.best_model)
            self.best_models_f1_test.append(objective.best_test_score)
            self.results_df = pd.concat(
                [self.results_df, objective.model_results_df],
                ignore_index=True
            )
