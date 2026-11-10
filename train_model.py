#!/usr/bin/env python3
"""
Train XGBoost World Cup prediction model for a specific cutoff date.

This script replicates the modeling pipeline from predictions.ipynb:
- Chronological train/calibration/test split (60/20/20)
- Optuna hyperparameter tuning with TimeSeriesSplit
- Temperature scaling calibration
- Export model artifacts as .pkl
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from scipy.special import softmax


# Class configuration
CLASS_ORDER = np.array([0, 1, 2], dtype=int)
CLASS_NAMES = {0: "Home win", 1: "Draw", 2: "Away win"}
CLASS_LABELS = [CLASS_NAMES[i] for i in CLASS_ORDER]


def validate_labels(y, split_name, require_all_classes=True):
    """Fail early when winner_code does not follow the agreed convention."""
    values = set(pd.Series(y).dropna().astype(int).unique())
    expected = set(CLASS_ORDER.tolist())

    unexpected = values - expected
    if unexpected:
        raise ValueError(
            f"{split_name} contains unexpected winner_code values: "
            f"{sorted(unexpected)}. Expected only {sorted(expected)}."
        )

    if require_all_classes:
        missing = expected - values
        if missing:
            raise ValueError(
                f"{split_name} is missing classes {sorted(missing)}. "
                "All three classes are required for model training."
            )


def align_predict_proba(model, raw_probs):
    """Return probability columns in fixed order: home, draw, away."""
    model_classes = np.asarray(model.classes_, dtype=int)
    missing = set(CLASS_ORDER.tolist()) - set(model_classes.tolist())
    if missing:
        raise ValueError(
            f"The fitted model is missing probability columns for classes {sorted(missing)}."
        )

    class_to_column = {
        class_id: column_index
        for column_index, class_id in enumerate(model_classes)
    }
    return np.column_stack(
        [raw_probs[:, class_to_column[class_id]] for class_id in CLASS_ORDER]
    )


def multiclass_brier(y_true, probs):
    y_true = np.asarray(y_true, dtype=int)
    validate_labels(y_true, "Brier-score labels")
    y_onehot = np.eye(len(CLASS_ORDER))[y_true]
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))


def rps_score_raw(y_true, probs):
    """Unnormalized cumulative RPS sum."""
    y_true = np.asarray(y_true, dtype=int)
    validate_labels(y_true, "RPS labels")
    actual = np.eye(len(CLASS_ORDER))[y_true]

    pred_cum = np.cumsum(probs[:, :-1], axis=1)
    actual_cum = np.cumsum(actual[:, :-1], axis=1)
    return np.mean(np.sum((pred_cum - actual_cum) ** 2, axis=1))


def rps_score(y_true, probs):
    """Standard normalized RPS for ordered outcomes home/draw/away."""
    return rps_score_raw(y_true, probs) / (len(CLASS_ORDER) - 1)


def apply_temperature(probabilities, temperature):
    """Apply multiclass temperature scaling."""
    probabilities = np.asarray(probabilities, dtype=float)

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    clipped = np.clip(probabilities, 1e-15, 1.0)
    log_probabilities = np.log(clipped)

    return softmax(log_probabilities / temperature, axis=1)


def fit_temperature(y_calibration, uncalibrated_probabilities):
    """Fit temperature using only the chronological calibration set."""

    def calibration_objective(log_temperature):
        temperature = np.exp(log_temperature)

        calibrated = apply_temperature(
            uncalibrated_probabilities,
            temperature,
        )

        return log_loss(
            y_calibration,
            calibrated,
            labels=CLASS_ORDER,
        )

    result = minimize_scalar(
        calibration_objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(
            f"Temperature optimization failed: {result.message}"
        )

    return float(np.exp(result.x))


def load_and_prepare_data(features_path: str):
    """Load features and prepare X, y."""
    print(f"Loading features from {features_path}...")
    features = pd.read_csv(features_path)
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    
    # Sort by date
    features = features.sort_values(by="date", ascending=True, kind="stable").reset_index(drop=True)
    
    print(f"Winner code distribution:")
    print(features["winner_code"].value_counts(dropna=False).sort_index())
    validate_labels(features["winner_code"], "Full dataset", require_all_classes=False)
    
    assert features["date"].is_monotonic_increasing, "The dataset is not sorted by date."
    
    # Keep date for splitting
    dates = features["date"].copy()
    
    # Drop non-feature columns (same as predictions.ipynb)
    drop_cols = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "winner",
        "tournament",
        "home_failed_score_rate_last_5",
        "away_failed_score_rate_last_5",
        "gap_bin",  # categorical, not used as feature
    ]
    
    # Check which columns exist
    existing_drop_cols = [c for c in drop_cols if c in features.columns]
    features = features.drop(columns=existing_drop_cols)
    
    y = features["winner_code"].astype(int)
    X = features.drop(columns=["winner_code"])
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    return X, y, dates


def chronological_split(X, y, train_ratio=0.6, calibration_ratio=0.2):
    """Split data chronologically into train, calibration, test."""
    n = len(X)
    train_end = int(n * train_ratio)
    calibration_end = int(n * (train_ratio + calibration_ratio))
    
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    
    X_cal = X.iloc[train_end:calibration_end].copy()
    y_cal = y.iloc[train_end:calibration_end].copy()
    
    X_test = X.iloc[calibration_end:].copy()
    y_test = y.iloc[calibration_end:].copy()
    
    validate_labels(y_train, "Development set", require_all_classes=True)
    validate_labels(y_cal, "Calibration set", require_all_classes=True)
    validate_labels(y_test, "Test set", require_all_classes=False)
    
    print(f"Development rows: {len(X_train)}")
    print(f"Calibration rows: {len(X_cal)}")
    print(f"Test rows: {len(X_test)}")
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def objective(trial, X_train, y_train):
    """Optuna objective function."""
    params = {
        "objective": "multi:softprob",
        "num_class": len(CLASS_ORDER),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", low=200, high=800, step=20),
        "learning_rate": trial.suggest_float("learning_rate", low=0.03, high=0.17, log=True),
        "max_depth": trial.suggest_int("max_depth", low=2, high=6, step=1),
        "min_child_weight": trial.suggest_int("min_child_weight", low=1, high=15, step=1),
        "gamma": trial.suggest_float("gamma", low=0.0, high=5.0, step=0.1),
        "subsample": trial.suggest_float("subsample", low=0.6, high=1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", low=0.6, high=1.0, step=0.05),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", low=0.6, high=1.0, step=0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", low=1.0, high=20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", low=0.0, high=5.0, step=0.1),
        "tree_method": "hist",
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=12)
    
    fold_rps_scores = []
    fold_brier_scores = []
    fold_log_losses = []

    for fold_number, (train_idx, val_idx) in enumerate(
        tscv.split(X_train),
        start=1,
    ):
        fold_X_train = X_train.iloc[train_idx]
        fold_X_val = X_train.iloc[val_idx]
        fold_y_train = y_train.iloc[train_idx]
        fold_y_val = y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            fold_X_train,
            fold_y_train,
            eval_set=[(fold_X_val, fold_y_val)],
            verbose=False,
        )

        raw_probs = model.predict_proba(fold_X_val)
        probs = align_predict_proba(model, raw_probs)

        fold_rps = rps_score(fold_y_val, probs)
        fold_brier = multiclass_brier(fold_y_val, probs)
        fold_logloss = log_loss(
            fold_y_val,
            probs,
            labels=CLASS_ORDER,
        )

        fold_rps_scores.append(fold_rps)
        fold_brier_scores.append(fold_brier)
        fold_log_losses.append(fold_logloss)

        running_rps = float(np.mean(fold_rps_scores))
        trial.report(running_rps, step=fold_number)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_rps = float(np.mean(fold_rps_scores))
    std_rps = float(np.std(fold_rps_scores))
    mean_brier = float(np.mean(fold_brier_scores))
    mean_logloss = float(np.mean(fold_log_losses))

    trial.set_user_attr("rps_std", std_rps)
    trial.set_user_attr("mean_brier", mean_brier)
    trial.set_user_attr("mean_logloss", mean_logloss)

    print(
        f"Trial: {trial.number} "
        f"- Mean RPS: {mean_rps:.5f} "
        f"- RPS std: {std_rps:.5f} "
        f"- Mean Brier: {mean_brier:.5f} "
        f"- Mean log loss: {mean_logloss:.5f}"
    )

    return mean_rps


def train_final_model(X_train, y_train, best_params):
    """Train final model on full training set with best parameters."""
    params = {
        "objective": "multi:softprob",
        "num_class": len(CLASS_ORDER),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        **best_params,
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    return model


def evaluate_model(model, X, y, name):
    """Evaluate model on a dataset."""
    raw_probs = model.predict_proba(X)
    probs = align_predict_proba(model, raw_probs)
    
    acc = np.mean(np.argmax(probs, axis=1) == y)
    ll = log_loss(y, probs, labels=CLASS_ORDER)
    brier = multiclass_brier(y, probs)
    rps = rps_score(y, probs)
    
    print(f"{name}: Accuracy={acc:.4f}, LogLoss={ll:.4f}, Brier={brier:.4f}, RPS={rps:.4f}")
    
    return {"accuracy": acc, "log_loss": ll, "brier": brier, "rps": rps}


def get_feature_importance(model, feature_names):
    """Get feature importance from trained model."""
    importance = model.feature_importances_
    return dict(zip(feature_names, importance))


def main():
    parser = argparse.ArgumentParser(description="Train World Cup prediction model")
    parser.add_argument("--features", type=str, required=True, help="Features CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output .pkl path")
    parser.add_argument("--cutoff", type=str, required=True, help="Cutoff date (YYYY-MM-DD)")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()
    
    # Load and prepare data
    X, y, dates = load_and_prepare_data(args.features)
    
    # Chronological split
    X_train, y_train, X_cal, y_cal, X_test, y_test = chronological_split(X, y)
    
    # Optuna optimization
    print(f"\nRunning Optuna optimization ({args.n_trials} trials)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2,
        ),
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    print(f"\nBest params: {study.best_params}")
    print(f"Best RPS: {study.best_value:.5f}")
    
    # Train final model on full development set (train + calibration)
    X_dev = pd.concat([X_train, X_cal])
    y_dev = pd.concat([y_train, y_cal])
    
    print("\nTraining final model on full development set...")
    final_model = train_final_model(X_dev, y_dev, study.best_params)
    
    # Calibrate on calibration set
    print("Calibrating with temperature scaling...")
    raw_probs_cal = final_model.predict_proba(X_cal)
    probs_cal = align_predict_proba(final_model, raw_probs_cal)
    temperature = fit_temperature(y_cal, probs_cal)
    print(f"Temperature: {temperature:.4f}")
    
    # Evaluate on test set (uncalibrated and calibrated)
    print("\n--- Test Set Evaluation (Uncalibrated) ---")
    test_metrics_uncal = evaluate_model(final_model, X_test, y_test, "Test")
    
    raw_probs_test = final_model.predict_proba(X_test)
    probs_test = align_predict_proba(final_model, raw_probs_test)
    calibrated_probs_test = apply_temperature(probs_test, temperature)
    
    print("\n--- Test Set Evaluation (Calibrated) ---")
    acc = np.mean(np.argmax(calibrated_probs_test, axis=1) == y_test)
    ll = log_loss(y_test, calibrated_probs_test, labels=CLASS_ORDER)
    brier = multiclass_brier(y_test, calibrated_probs_test)
    rps = rps_score(y_test, calibrated_probs_test)
    print(f"Test: Accuracy={acc:.4f}, LogLoss={ll:.4f}, Brier={brier:.4f}, RPS={rps:.4f}")
    test_metrics_cal = {"accuracy": acc, "log_loss": ll, "brier": brier, "rps": rps}
    
    # Also evaluate on calibration set
    print("\n--- Calibration Set Evaluation (Calibrated) ---")
    calibrated_probs_cal = apply_temperature(probs_cal, temperature)
    acc_cal = np.mean(np.argmax(calibrated_probs_cal, axis=1) == y_cal)
    ll_cal = log_loss(y_cal, calibrated_probs_cal, labels=CLASS_ORDER)
    brier_cal = multiclass_brier(y_cal, calibrated_probs_cal)
    rps_cal = rps_score(y_cal, calibrated_probs_cal)
    print(f"Calibration: Accuracy={acc_cal:.4f}, LogLoss={ll_cal:.4f}, Brier={brier_cal:.4f}, RPS={rps_cal:.4f}")
    
    # Feature importance
    feature_importance = get_feature_importance(final_model, X.columns.tolist())
    
    # Prepare artifacts for export
    artifacts = {
        "model": final_model,
        "temperature": temperature,
        "feature_names": X.columns.tolist(),
        "feature_importance": feature_importance,
        "best_params": study.best_params,
        "optuna_study": study,
        "cutoff_date": args.cutoff,
        "train_size": len(X_train),
        "calibration_size": len(X_cal),
        "test_size": len(X_test),
        "test_metrics_uncalibrated": test_metrics_uncal,
        "test_metrics_calibrated": test_metrics_cal,
        "calibration_metrics_calibrated": {
            "accuracy": acc_cal,
            "log_loss": ll_cal,
            "brier": brier_cal,
            "rps": rps_cal,
        },
        "class_order": CLASS_ORDER.tolist(),
        "class_names": CLASS_NAMES,
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()