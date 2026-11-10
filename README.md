# World Cup Prediction Modeling

1. Feature Generation (generate_features.py)
- Generates features for any cutoff date using precomputed stored features
- Adds historical upset feature: P(lower Pi rating team wins) stratified by Pi rating gap bins
- Created feature files for three cutoffs:
- features/features_2018.csv (33,765 matches, cutoff 2018-06-14)
- features/features_2022.csv (37,646 matches, cutoff 2022-11-20)
- features/features_2026.csv (41,162 matches, cutoff 2026-06-11)

2. Model Training (train_model.py)
- XGBoost multiclass (home/draw/away) with Optuna hyperparameter tuning
- Chronological train/calibration/test split (60/20/20)
- Temperature scaling calibration
- Exports .pkl with model, calibrator, feature names, importance, metrics
- Trained three models:
- models/wc2018_model.pkl - Test LogLoss: 0.9190 (calibrated)
- models/wc2022_model.pkl - Test LogLoss: 0.8986 (calibrated)
- models/wc2026_model.pkl - Test LogLoss: 0.8879 (calibrated)

3. Baseline Comparison (baseline_models.py)
- Multinomial Logistic Regression and Random Forest
- Results (Test LogLoss):
- 2018: LR=0.9183, RF=0.9403 (XGBoost: 0.9190)
- 2022: LR=0.9002, RF=0.9260 (XGBoost: 0.8986)
- 2026: LR=0.8900, RF=0.9150 (XGBoost: 0.8879)
- XGBoost outperforms both baselines on all cutoffs

4. Monte Carlo Simulation (monte_carlo.py)
- Generates features for WC matches using team history up to cutoff
- Runs simulations with calibrated probabilities
- 1000 simulations per World Cup
- 2018: Belgium 35.5%, France 31.3%, England 16.8% (Actual: France)
- 2022: Argentina 35.4%, Croatia 30.1%, France 8.4% (Actual: Argentina)
- 2026: Belgium 7.2%, Ecuador 6.8%, Algeria 6.1%