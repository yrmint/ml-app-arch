from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import optuna
import librosa
import pandas as pd
import soundfile as sf
import warnings
import torch.nn.functional as F
import torch


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
            channels = tuple(first_channel * (2 ** i) for i in range (0, n_channels))
            clf_obj = SciKitCNN(lr=1e-4, batch_size=batch_size, channels=channels,
                                     dropout=dropout, epoch_n=epoch_n)
        else:
            raise ValueError(f"Unknown model: {clf_name}")

	
		# A little bit of logging
        print(f"\n{clf_name} hyperoptimization with params: lr {1e-4}, channels {channels}, dropout {dropout}")

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
        self.model_results_df = pd.concat([self.model_results_df, pd.DataFrame({
            'n': trial.number,
            'Model': clf_name,
            'Parameters': [trial.params],
            'F1_val': f1_val,
            'F1_test': f1_test
        })], ignore_index=True)

        return f1_val


class ModelOptimization:
    """
    Class for hyperparam search
    As the result - DF with logs
    """

    def __init__(self, models_list):
        self.models_list = models_list
        self.results_df = pd.DataFrame(columns=['n', 'Model', 'F1_val', 'F1_test', 'Parameters'])
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_f1_test = []
        self.studies = defaultdict(lambda: [])

    def fit(self, x, y, X_val, y_val, X_test, y_test, n_trials=20, n_startup_trials=10):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        model = self.models_list[0]
        for s in ['tpe']:
            print(f"\n{model} hyperoptimization")

            if s == 'tpe':
                sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=n_startup_trials)
            elif s == 'cmaes':
                sampler = optuna.samplers.CmaEsSampler(lr_adapt=True, n_startup_trials=n_startup_trials)

            objective = Objective(x, y, model, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
            study = optuna.create_study(direction="maximize", sampler=sampler, study_name="CNN_optimization",
                                        load_if_exists=True, storage="sqlite:///db.sqlite3")
            study.set_metric_names(["F1_val"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            self.best_models.append(objective.best_model)
            self.best_models_f1_test.append(objective.best_test_score)
            self.results_df = pd.concat([self.results_df, objective.model_results_df], ignore_index=True)
            

def preprocess_audio(file_path: Union[str, bytes, Path]) -> torch.Tensor:
    """
    Returns tensor for SciKitCNN model inference:
    shape = (n_segments, 1, image_size, image_size)
    """
    target_sr = 22050
    segment_length = 4
    overlap = 2
    
    n_fft = 2048
    hop_length = 512
    n_mels = 150
    image_size = 150
    
	# 
    #audio_file = BytesIO(file_path)
    
    if isinstance(file_path, (str, Path)):
        audio_file = str(file_path)
    elif isinstance(file_path, bytes):
        audio_file = BytesIO(file_path)
    else:
        raise TypeError(f"Path or bytes are only valid optins, input type: {type(file_path)}")
    
    #audio_file = BytesIO(file_path) #if file_path is bytes else file_path
    
    audio, file_sr = librosa.load(audio_file, sr=target_sr, mono=True)
    
    #audio, file_sr = sf.read(audio_file)   
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)

    segment_samples = segment_length * target_sr
    step = (segment_length - overlap) * target_sr
    
    segments = []
    
    if len(audio) < segment_samples:
        assert Exception("Too shirt audio length")

    for start in range(0, len(audio) - segment_samples + 1, step):
        end = start + segment_samples
        segments.append(audio[start:end])

    mel_specs = []
    for segment in segments:
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=target_sr, n_fft=n_fft,
                                                  hop_length=hop_length, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        

        #min_db = -80.0  # стандартный порог тишины
        #mel_spec_norm = np.clip((mel_spec_db - min_db) / (-min_db), 0, 1)
        
        # Transform into a tensor for interpolation: [Batch=1, Channel=1, H, W]
        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Resize to (150, 150)
        mel_resized = F.interpolate(mel_tensor, size=(image_size, image_size),
                                    mode='bilinear', align_corners=False)
        
        # Make shape [1, 150, 150]
        mel_specs.append(mel_resized.squeeze(0))

    output_tensor = torch.stack(mel_specs)
    
    return output_tensor
    