#!/usr/bin/env python3
"""
Monte Carlo simulation for World Cup tournaments.

Loads trained model, generates features for WC matches using team history up to cutoff,
runs simulations to estimate team probabilities.
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.special import softmax


# Class configuration
CLASS_ORDER = np.array([0, 1, 2], dtype=int)
CLASS_NAMES = {0: "Home win", 1: "Draw", 2: "Away win"}


def apply_temperature(probabilities, temperature):
    """Apply multiclass temperature scaling."""
    probabilities = np.asarray(probabilities, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    clipped = np.clip(probabilities, 1e-15, 1.0)
    log_probabilities = np.log(clipped)
    return softmax(log_probabilities / temperature, axis=1)


def align_predict_proba(model, raw_probs):
    """Return probability columns in fixed order: home, draw, away."""
    model_classes = np.asarray(model.classes_, dtype=int)
    class_to_column = {
        class_id: column_index
        for column_index, class_id in enumerate(model_classes)
    }
    return np.column_stack(
        [raw_probs[:, class_to_column[class_id]] for class_id in CLASS_ORDER]
    )


def load_model(model_path: str):
    """Load trained model artifacts."""
    print(f"Loading model from {model_path}...")
    artifacts = joblib.load(model_path)
    return artifacts


def load_raw_wc_matches(cutoff_year: int):
    """Load raw World Cup matches from international_football_games.csv or wc_26.csv."""
    if cutoff_year == 2026:
        wc_df = pd.read_csv("data/wc2026dataset/wc_26.csv")
        wc_df["date"] = pd.to_datetime(wc_df["date"])
        wc_df = wc_df[wc_df["tournament"] == "FIFA World Cup"].copy()
        wc_df["match_id"] = (
            wc_df["date"].astype(str)
            + "_"
            + wc_df["home_team"].astype(str).str.replace(" ", "_")
            + "_"
            + wc_df["away_team"].astype(str).str.replace(" ", "_")
            + "_0"
        )
        return wc_df
    else:
        games_df = pd.read_csv("data/international_football_games.csv")
        games_df["date"] = pd.to_datetime(games_df["date"])
        
        if cutoff_year == 2018:
            wc_start = pd.Timestamp("2018-06-14")
            wc_end = pd.Timestamp("2018-07-15")
        elif cutoff_year == 2022:
            wc_start = pd.Timestamp("2022-11-20")
            wc_end = pd.Timestamp("2022-12-18")
        else:
            wc_start = pd.Timestamp(f"{cutoff_year}-01-01")
            wc_end = pd.Timestamp(f"{cutoff_year}-12-31")
        
        wc_df = games_df[
            (games_df["tournament"] == "FIFA World Cup") &
            (games_df["date"] >= wc_start) &
            (games_df["date"] <= wc_end)
        ].copy()
        
        wc_df["_match_occurrence"] = (
            wc_df.groupby(["date", "home_team", "away_team"], dropna=False).cumcount()
        )
        wc_df["match_id"] = (
            wc_df["date"].astype(str)
            + "_"
            + wc_df["home_team"].astype(str).str.replace(" ", "_")
            + "_"
            + wc_df["away_team"].astype(str).str.replace(" ", "_")
            + "_"
            + wc_df["_match_occurrence"].astype(str)
        )
        wc_df.drop(columns="_match_occurrence", inplace=True)
        
        return wc_df


def get_team_features_for_match(team_history, team, cutoff_date):
    """Calculate team features for a match using history up to cutoff_date."""
    TEAM_FEATURE_COLUMNS = [
        "team", "date", "ppg_last_5", "ppg_last_10", "avg_goals_scored_last_5",
        "avg_goals_conceded_last_5", "clean_sheets_rate_last_5",
        "avg_goal_difference_last_5", "avg_goal_difference_last_10",
        "btts_rate_last_5", "failed_score_rate_last_5", "over_2_5_rate_last_5",
        "over_3_5_rate_last_5", "under_1_5_rate_last_5", "days_since_last_game",
        "goal_difference_trend", "momentum", "pi_home_rating", "pi_away_rating",
        "pi_neutral_rating", "pi_expected_gd", "pi_diff",
    ]
    
    _BASE_REQUIRED_COLUMNS = {"date", "team", "score", "opponent_score"}
    _RATING_COLUMNS = [
        "pi_home_rating", "pi_away_rating", "pi_neutral_rating",
        "pi_expected_gd", "pi_diff",
    ]
    
    def _mean_last(series, n):
        values = pd.to_numeric(series, errors="coerce").dropna().tail(n)
        return float(values.mean()) if not values.empty else np.nan
    
    df = team_history[team_history["team"] == team].sort_values("date")
    
    if df.empty:
        return {col: np.nan for col in TEAM_FEATURE_COLUMNS} | {"team": team, "date": cutoff_date}
    
    hist = df[df["date"] < cutoff_date]
    if hist.empty:
        return {col: np.nan for col in TEAM_FEATURE_COLUMNS} | {"team": team, "date": cutoff_date}
    
    if "points" not in hist.columns:
        hist["points"] = np.select(
            [hist["score"] > hist["opponent_score"], hist["score"] == hist["opponent_score"]],
            [3, 1], default=0
        )
    
    score = pd.to_numeric(hist["score"], errors="coerce")
    conceded = pd.to_numeric(hist["opponent_score"], errors="coerce")
    gd = score - conceded
    total = score + conceded
    
    result = {
        "team": team, "date": cutoff_date,
        "ppg_last_5": _mean_last(hist["points"], 5),
        "ppg_last_10": _mean_last(hist["points"], 10),
        "avg_goals_scored_last_5": _mean_last(score, 5),
        "avg_goals_conceded_last_5": _mean_last(conceded, 5),
        "clean_sheets_rate_last_5": _mean_last(conceded.eq(0).astype(float), 5),
        "avg_goal_difference_last_5": _mean_last(gd, 5),
        "avg_goal_difference_last_10": _mean_last(gd, 10),
        "btts_rate_last_5": _mean_last(((score > 0) & (conceded > 0)).astype(float), 5),
        "failed_score_rate_last_5": _mean_last(score.eq(0).astype(float), 5),
        "over_2_5_rate_last_5": _mean_last(total.gt(2.5).astype(float), 5),
        "over_3_5_rate_last_5": _mean_last(total.gt(3.5).astype(float), 5),
        "under_1_5_rate_last_5": _mean_last(total.lt(1.5).astype(float), 5),
        "days_since_last_game": max((cutoff_date - hist["date"].max()).days, 0),
        "goal_difference_trend": _mean_last(gd, 5) - _mean_last(gd, 10),
        "momentum": _mean_last(hist["points"], 5) - _mean_last(hist["points"], 10),
    }
    
    rating_rows = df[df["date"] <= cutoff_date].sort_values("date")
    if not rating_rows.empty:
        rr = rating_rows.iloc[-1]
        for c in _RATING_COLUMNS:
            result[c] = rr[c] if c in rr.index else np.nan
    else:
        for c in _RATING_COLUMNS:
            result[c] = np.nan
    
    return result


def create_match_features(home_feats, away_feats, neutral, historical_upset_rate=0.22):
    """Create match features from two team feature sets."""
    TEAM_FEATURE_COLUMNS = [
        "team", "date", "ppg_last_5", "ppg_last_10", "avg_goals_scored_last_5",
        "avg_goals_conceded_last_5", "clean_sheets_rate_last_5",
        "avg_goal_difference_last_5", "avg_goal_difference_last_10",
        "btts_rate_last_5", "failed_score_rate_last_5", "over_2_5_rate_last_5",
        "over_3_5_rate_last_5", "under_1_5_rate_last_5", "days_since_last_game",
        "goal_difference_trend", "momentum", "pi_home_rating", "pi_away_rating",
        "pi_neutral_rating", "pi_expected_gd", "pi_diff",
    ]
    
    def safe_ratio(num, den):
        num, den = float(num), float(den)
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return 0.0
        v = num / den
        return v if np.isfinite(v) else 0.0
    
    h = home_feats
    a = away_feats
    
    row = {"home_team": h["team"], "away_team": a["team"], "home_neutral": int(neutral)}
    for c in [c for c in TEAM_FEATURE_COLUMNS if c not in {"team", "date"}]:
        row[f"home_{c}"] = h.get(c, np.nan)
        row[f"away_{c}"] = a.get(c, np.nan)
    
    if neutral:
        row["pi_rating_diff"] = h["pi_away_rating"] - a["pi_away_rating"]
    else:
        row["pi_rating_diff"] = h["pi_home_rating"] - a["pi_away_rating"]
    
    row["diff_ppg_last_5"] = h["ppg_last_5"] - a["ppg_last_5"]
    row["diff_goal_difference"] = h["avg_goal_difference_last_5"] - a["avg_goal_difference_last_5"]
    row["diff_btts_rate"] = h["btts_rate_last_5"] - a["btts_rate_last_5"]
    row["diff_avg_goals_scored_last_5"] = h["avg_goals_scored_last_5"] - a["avg_goals_scored_last_5"]
    row["diff_avg_goals_conceded_last_5"] = a["avg_goals_conceded_last_5"] - h["avg_goals_conceded_last_5"]
    row["diff_clean_sheets_rate"] = h["clean_sheets_rate_last_5"] - a["clean_sheets_rate_last_5"]
    row["diff_over_2_5_rate"] = h["over_2_5_rate_last_5"] - a["over_2_5_rate_last_5"]
    row["diff_under_1_5_rate"] = h["under_1_5_rate_last_5"] - a["under_1_5_rate_last_5"]
    row["diff_rests_days"] = h["days_since_last_game"] - a["days_since_last_game"]
    row["expected_gd_edge"] = h["pi_expected_gd"] - a["pi_expected_gd"]
    row["diff_momentum"] = h["momentum"] - a["momentum"]
    row["home_pi_momentum"] = h["pi_home_rating"] * h["momentum"]
    row["away_pi_momentum"] = a["pi_away_rating"] * a["momentum"]
    row["home_weighted_avg_ppg"] = 0.7 * h["ppg_last_5"] + 0.3 * h["ppg_last_10"]
    row["away_weighted_avg_ppg"] = 0.7 * a["ppg_last_5"] + 0.3 * a["ppg_last_10"]
    row["home_goal_to_concede_rate"] = safe_ratio(h["avg_goals_scored_last_5"], h["avg_goals_conceded_last_5"])
    row["away_goal_to_concede_rate"] = safe_ratio(a["avg_goals_scored_last_5"], a["avg_goals_conceded_last_5"])
    row["historical_upset_rate"] = historical_upset_rate
    
    return row


def generate_wc_match_features(cutoff_year, cutoff_date, model_artifacts):
    """Generate features for all WC matches using team history up to cutoff."""
    print(f"Generating features for {cutoff_year} WC matches...")
    
    # Load raw WC matches
    wc_matches = load_raw_wc_matches(cutoff_year)
    
    # Load team history up to cutoff
    team_features = pd.read_csv("stored_features/team_features_c_25_mu1_0.20_mu2_0.10.csv")
    team_features["date"] = pd.to_datetime(team_features["date"])
    team_history = team_features[team_features["date"] < cutoff_date].copy()
    
    # Global historical upset rate (approximate)
    historical_upset_rate = 0.22
    
    match_features = []
    for _, m in wc_matches.iterrows():
        home_feats = get_team_features_for_match(team_history, m["home_team"], cutoff_date)
        away_feats = get_team_features_for_match(team_history, m["away_team"], cutoff_date)
        
        feat = create_match_features(home_feats, away_feats, m["neutral"], historical_upset_rate)
        feat["date"] = m["date"]
        feat["match_id"] = m["match_id"]
        feat["tournament"] = "FIFA World Cup"
        feat["home_score"] = m.get("home_score", np.nan)
        feat["away_score"] = m.get("away_score", np.nan)
        feat["winner"] = m.get("winner", np.nan)
        feat["winner_code"] = m.get("winner_code", np.nan)
        match_features.append(feat)
    
    wc_features = pd.DataFrame(match_features)
    wc_features = wc_features.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"Generated features for {len(wc_features)} WC matches")
    
    return wc_features


def simulate_tournament(wc_features, model, feature_names, temperature, rng):
    """Simulate a single World Cup tournament (simplified)."""
    teams = set()
    for _, m in wc_features.iterrows():
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    teams = sorted(list(teams))
    
    # Initialize
    points = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    
    # Simulate all matches
    for _, match in wc_features.iterrows():
        X = pd.DataFrame([match])[feature_names].fillna(0)
        raw_probs = model.predict_proba(X)[0]
        probs = align_predict_proba(model, raw_probs.reshape(1, -1))[0]
        calibrated = apply_temperature(probs.reshape(1, -1), temperature)[0]
        
        outcome = rng.choice(3, p=calibrated)
        home, away = match["home_team"], match["away_team"]
        
        if outcome == 0:  # Home win
            points[home] += 3
            gf[home] += rng.poisson(1.5) + 1
            ga[away] += rng.poisson(0.5)
        elif outcome == 2:  # Away win
            points[away] += 3
            gf[away] += rng.poisson(1.5) + 1
            ga[home] += rng.poisson(0.5)
        else:  # Draw
            points[home] += 1
            points[away] += 1
            goals = rng.poisson(1.0)
            gf[home] += goals
            ga[home] += goals
            gf[away] += goals
            ga[away] += goals
    
    # Simple ranking by points, then GD
    ranking = sorted(teams, key=lambda t: (points[t], gf[t] - ga[t], gf[t]), reverse=True)
    
    winner = ranking[0]
    runner_up = ranking[1]
    semi_finalists = ranking[:4]
    quarter_finalists = ranking[:8]
    round_16 = ranking[:16]
    
    return {
        "winner": winner,
        "runner_up": runner_up,
        "semi_finalists": semi_finalists,
        "quarter_finalists": quarter_finalists,
        "round_16": round_16,
        "points": points,
    }


def run_monte_carlo(
    model_artifacts,
    wc_features,
    n_simulations=10000,
    seed=42,
):
    """Run Monte Carlo simulations."""
    rng = np.random.default_rng(seed)
    
    model = model_artifacts["model"]
    temperature = model_artifacts["temperature"]
    feature_names = model_artifacts["feature_names"]
    
    teams = set()
    for _, m in wc_features.iterrows():
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    teams = sorted(list(teams))
    
    print(f"Running {n_simulations} simulations for {len(teams)} teams...")
    
    # Accumulators
    win_count = {t: 0 for t in teams}
    stage_count = {t: {"R16": 0, "QF": 0, "SF": 0, "F": 0, "W": 0} for t in teams}
    
    for i in range(n_simulations):
        if i % 1000 == 0:
            print(f"  Simulation {i}/{n_simulations}")
        
        result = simulate_tournament(wc_features, model, feature_names, temperature, rng)
        
        win_count[result["winner"]] += 1
        stage_count[result["winner"]]["W"] += 1
        stage_count[result["runner_up"]]["F"] += 1
        for t in result["semi_finalists"]:
            stage_count[t]["SF"] += 1
        for t in result["quarter_finalists"]:
            stage_count[t]["QF"] += 1
        for t in result["round_16"]:
            stage_count[t]["R16"] += 1
    
    # Convert to probabilities
    results = {
        "team_win_probability": {t: win_count[t] / n_simulations for t in teams},
        "team_stage_probability": {
            t: {stage: stage_count[t][stage] / n_simulations for stage in ["R16", "QF", "SF", "F", "W"]}
            for t in teams
        },
        "n_simulations": n_simulations,
        "model_cutoff": model_artifacts.get("cutoff_date", "unknown"),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation for World Cup")
    parser.add_argument("--model", type=str, required=True, help="Model .pkl path")
    parser.add_argument("--cutoff", type=str, required=True, help="Cutoff year (2018, 2022, 2026)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    cutoff_year = int(args.cutoff)
    
    # Load model
    model_artifacts = load_model(args.model)
    
    # Determine cutoff date
    if cutoff_year == 2018:
        cutoff_date = pd.Timestamp("2018-06-14")
    elif cutoff_year == 2022:
        cutoff_date = pd.Timestamp("2022-11-20")
    else:
        cutoff_date = pd.Timestamp("2026-06-11")
    
    # Generate WC match features
    wc_features = generate_wc_match_features(cutoff_year, cutoff_date, model_artifacts)
    
    if len(wc_features) == 0:
        print("No WC matches found!")
        return
    
    # Run simulation
    results = run_monte_carlo(
        model_artifacts,
        wc_features,
        n_simulations=args.n_sims,
        seed=args.seed,
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Print top teams
    print("\n=== TOP 10 WIN PROBABILITIES ===")
    sorted_teams = sorted(
        results["team_win_probability"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for team, prob in sorted_teams[:10]:
        print(f"  {team}: {prob:.4f}")


if __name__ == "__main__":
    main()