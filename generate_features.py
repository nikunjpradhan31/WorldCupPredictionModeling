#!/usr/bin/env python3
"""
Generate match features for World Cup prediction models at specific cutoff dates.

This script uses the precomputed stored_features and adds the historical upset feature.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# Pi rating parameters (from stored_features filenames)
PI_C = 25.0
PI_MU1 = 0.20
PI_MU2 = 0.10


def compute_historical_upset_feature(
    games_df: pd.DataFrame,
    team_features_pi: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    pi_gap_bins: list = None,
) -> pd.DataFrame:
    """
    Compute historical upset probability: P(lower Pi rating team wins | match).
    
    For each match, we look at all historical matches up to that match's date
    and compute the empirical probability that the team with lower Pi rating wins.
    
    Can be stratified by Pi rating gap bins.
    """
    if pi_gap_bins is None:
        pi_gap_bins = [-np.inf, 50, 100, 200, 500, np.inf]
        pi_gap_labels = ['0-50', '50-100', '100-200', '200-500', '500+']
    
    # Sort by date
    games_df = games_df.sort_values("date", kind="stable").reset_index(drop=True)
    
    # Prepare match-level data with Pi ratings
    match_pi_data = []
    for match_id in games_df["match_id"].unique():
        match_rows = team_features_pi[team_features_pi["match_id"] == match_id]
        if len(match_rows) != 2:
            continue
        
        row_a = match_rows.iloc[0]
        row_b = match_rows.iloc[1]
        
        # Get match info from games_df
        match_info = games_df[games_df["match_id"] == match_id].iloc[0]
        
        if match_info["neutral"]:
            pi_home = row_a["pi_neutral_rating"] if row_a["is_home"] == 1 else row_b["pi_neutral_rating"]
            pi_away = row_b["pi_neutral_rating"] if row_b["is_home"] == 1 else row_a["pi_neutral_rating"]
        else:
            pi_home = row_a["pi_home_rating"] if row_a["is_home"] == 1 else row_b["pi_home_rating"]
            pi_away = row_b["pi_away_rating"] if row_b["is_home"] == 1 else row_a["pi_away_rating"]
        
        pi_diff = abs(pi_home - pi_away)
        lower_pi_team = match_info["home_team"] if pi_home < pi_away else match_info["away_team"]
        higher_pi_team = match_info["away_team"] if pi_home < pi_away else match_info["home_team"]
        
        # Determine winner
        if match_info["winner"] == "DRAW":
            winner = "DRAW"
        else:
            winner = match_info["winner"]
        
        lower_pi_won = 1 if winner == lower_pi_team else 0
        
        match_pi_data.append({
            "match_id": match_id,
            "date": match_info["date"],
            "pi_diff": pi_diff,
            "lower_pi_team": lower_pi_team,
            "higher_pi_team": higher_pi_team,
            "winner": winner,
            "lower_pi_won": lower_pi_won,
        })
    
    match_pi_df = pd.DataFrame(match_pi_data)
    match_pi_df["date"] = pd.to_datetime(match_pi_df["date"])
    match_pi_df = match_pi_df.sort_values("date", kind="stable").reset_index(drop=True)
    
    # Assign gap bins
    match_pi_df["gap_bin"] = pd.cut(
        match_pi_df["pi_diff"], 
        bins=pi_gap_bins, 
        labels=pi_gap_labels,
        include_lowest=True
    )
    
    # Compute cumulative upset rates per bin
    # For each match, compute the historical rate from ALL previous matches in same bin
    match_pi_df["historical_upset_rate"] = np.nan
    
    for bin_label in pi_gap_labels:
        bin_matches = match_pi_df[match_pi_df["gap_bin"] == bin_label].copy()
        if len(bin_matches) == 0:
            continue
        
        # Compute expanding mean of lower_pi_won (excluding current match)
        bin_matches = bin_matches.reset_index(drop=True)
        expanding_rates = bin_matches["lower_pi_won"].expanding().mean().shift(1)
        
        # Map back to original dataframe
        match_pi_df.loc[bin_matches.index, "historical_upset_rate"] = expanding_rates.values
    
    # Fill NaN with global average
    global_rate = match_pi_df["lower_pi_won"].expanding().mean().shift(1)
    match_pi_df["historical_upset_rate"] = match_pi_df["historical_upset_rate"].fillna(global_rate)
    match_pi_df["historical_upset_rate"] = match_pi_df["historical_upset_rate"].fillna(0.0)
    
    return match_pi_df[["match_id", "historical_upset_rate", "gap_bin"]]


def generate_features_for_cutoff(
    cutoff_date: str,
    output_path: str,
    start_date: str = "1960-01-01",
    stored_features_path: str = "stored_features/match_features_c_25_mu1_0.20_mu2_0.10.csv",
    games_csv: str = "data/international_football_games.csv",
    valid_teams_csv: str = "data/valid_fifa_teams.csv",
    shootouts_csv: str = "data/shootouts.csv",
):
    """
    Generate features for all matches up to a cutoff date using precomputed stored features.
    
    Parameters
    ----------
    cutoff_date : str
        Cutoff date in YYYY-MM-DD format. Only matches strictly before this date are used.
    output_path : str
        Path to save the output CSV.
    start_date : str
        Minimum date for matches to include (default 1960-01-01).
    """
    cutoff = pd.Timestamp(cutoff_date)
    start = pd.Timestamp(start_date)
    
    print(f"Loading stored features...")
    match_features = pd.read_csv(stored_features_path)
    match_features["date"] = pd.to_datetime(match_features["date"])
    
    print(f"Loading raw games for historical upset calculation...")
    games_df = pd.read_csv(games_csv)
    valid_teams_df = pd.read_csv(valid_teams_csv)
    draws_df = pd.read_csv(shootouts_csv)
    
    # Filter to valid teams
    teams = set(valid_teams_df["team"].dropna().unique())
    valid_team_mask = (
        games_df["home_team"].isin(teams)
        & games_df["away_team"].isin(teams)
    )
    games_df = games_df.loc[valid_team_mask].copy()
    
    # Merge shootouts
    games_df = games_df.merge(
        draws_df, 
        how='left', 
        left_on=['home_team', 'away_team', 'date'], 
        right_on=['home_team', 'away_team', 'date']
    ).drop(columns=['first_shooter'])
    
    games_df["date"] = pd.to_datetime(games_df["date"])
    
    # Determine winner
    games_df.loc[games_df["home_score"] > games_df["away_score"], "winner"] = games_df["home_team"]
    games_df.loc[games_df["home_score"] < games_df["away_score"], "winner"] = games_df["away_team"]
    games_df.loc[
        (games_df["home_score"] == games_df["away_score"]) & (games_df["winner"].isna()),
        "winner"
    ] = "DRAW"
    
    games_df["winner_code"] = np.select(
        [
            games_df["winner"] == games_df["home_team"],
            games_df["winner"] == games_df["away_team"],
            games_df["winner"] == "DRAW",
        ],
        [0, 2, 1],
        default=-1,
    )
    
    games_df["_match_occurrence"] = (
        games_df.groupby(["date", "home_team", "away_team"], dropna=False).cumcount()
    )
    games_df["match_id"] = (
        games_df["date"].astype(str)
        + "_"
        + games_df["home_team"].astype(str).str.replace(" ", "_")
        + "_"
        + games_df["away_team"].astype(str).str.replace(" ", "_")
        + "_"
        + games_df["_match_occurrence"].astype(str)
    )
    games_df.drop(columns="_match_occurrence", inplace=True)
    
    # Filter by date range for games_df (for historical upset calc)
    games_df = games_df[
        (games_df["date"] < cutoff) & (games_df["date"] >= start)
    ].sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    
    print(f"Matches in range [{start_date}, {cutoff_date}): {len(games_df)}")
    
    # Filter match_features by date range
    match_features = match_features[
        (match_features["date"] < cutoff) & (match_features["date"] >= start)
    ].sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    
    print(f"Stored features in range: {len(match_features)}")
    
    # We need team_features_per_date with Pi ratings for historical upset calculation
    # Load the team_features file
    print("Loading team features for Pi ratings...")
    team_features = pd.read_csv("stored_features/team_features_c_25_mu1_0.20_mu2_0.10.csv")
    team_features["date"] = pd.to_datetime(team_features["date"])
    team_features = team_features[
        (team_features["date"] < cutoff) & (team_features["date"] >= start)
    ].sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    
    print("Computing historical upset feature...")
    # Compute historical upset feature
    upset_df = compute_historical_upset_feature(
        games_df,
        team_features,
        cutoff,
    )
    
    # Merge upset feature back to match_features
    match_features = match_features.merge(upset_df, on="match_id", how="left")
    
    # Fill any missing upset rates with 0
    match_features["historical_upset_rate"] = match_features["historical_upset_rate"].fillna(0.0)
    match_features["gap_bin"] = match_features["gap_bin"].astype(str).fillna("unknown")
    
    print(f"Generated {len(match_features)} match features with historical upset rate")
    print(f"Historical upset rate stats: mean={match_features['historical_upset_rate'].mean():.4f}, std={match_features['historical_upset_rate'].std():.4f}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_features.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return match_features


def main():
    parser = argparse.ArgumentParser(description="Generate features for World Cup prediction models")
    parser.add_argument("--cutoff", type=str, required=True, help="Cutoff date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--start", type=str, default="1960-01-01", help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    generate_features_for_cutoff(args.cutoff, args.output, args.start)


if __name__ == "__main__":
    main()