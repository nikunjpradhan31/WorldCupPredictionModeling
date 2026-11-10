from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


TEAM_FEATURE_COLUMNS = [
    "team",
    "date",
    "ppg_last_5",
    "ppg_last_10",
    "avg_goals_scored_last_5",
    "avg_goals_conceded_last_5",
    "clean_sheets_rate_last_5",
    "avg_goal_difference_last_5",
    "avg_goal_difference_last_10",
    "btts_rate_last_5",
    "failed_score_rate_last_5",
    "over_2_5_rate_last_5",
    "over_3_5_rate_last_5",
    "under_1_5_rate_last_5",
    "days_since_last_game",
    "goal_difference_trend",
    "momentum",
    "pi_home_rating",
    "pi_away_rating",
    "pi_neutral_rating",
    "pi_expected_gd",
    "pi_diff",
]

_BASE_REQUIRED_COLUMNS = {
    "date",
    "team",
    "score",
    "opponent_score",
}

_RATING_COLUMNS = [
    "pi_home_rating",
    "pi_away_rating",
    "pi_neutral_rating",
    "pi_expected_gd",
    "pi_diff",
]


def _mean_last(series: pd.Series, n: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(n)
    return float(values.mean()) if not values.empty else np.nan


def _as_record(features: pd.Series | Mapping[str, Any] | pd.DataFrame) -> dict[str, Any]:
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("A team feature DataFrame must contain exactly one row.")
        return features.iloc[0].to_dict()
    if isinstance(features, pd.Series):
        return features.to_dict()
    if isinstance(features, Mapping):
        return dict(features)
    raise TypeError("Team features must be a mapping, pandas Series, or one-row DataFrame.")


def get_team_features(
    team_history: pd.DataFrame,
    team: str,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    latest: bool = True,
) -> pd.DataFrame:
    """Calculate one team's features for a prediction cutoff.

    Parameters
    ----------
    team_history:
        Team-per-match data, such as ``team_features`` from the notebook. It
        must contain ``date``, ``team``, ``score``, and ``opponent_score``.
        For exact notebook PPG, it should also contain ``points``; otherwise
        points are inferred from score and opponent_score.

        Pi columns are optional. When present, the most recent stored rating
        record on or before the cutoff is copied into the result.

    team:
        Team name exactly as it appears in ``team_history['team']``.

    start_date, end_date:
        Optional inclusive historical window. ``end_date`` is the as-of date.
        Only completed games strictly before this cutoff are used in rolling
        calculations. ``start_date`` limits how far back the calculations can
        look.

    latest:
        If True (default), use the latest date available for this team and
        ignore ``end_date``. If False, ``end_date`` is required.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame containing the team features used by
        ``create_match_features``.

    Notes
    -----
    The notebook stores Pi ratings as pre-match values. This helper therefore
    copies the latest stored pre-match rating record available by the cutoff.
    For a truly post-match current Pi state, persist the final rating state
    from ``add_pi_ratings`` separately.
    """
    missing = _BASE_REQUIRED_COLUMNS.difference(team_history.columns)
    if missing:
        raise ValueError(f"team_history is missing required columns: {sorted(missing)}")

    df = team_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.loc[df["team"].eq(team)].sort_values(["date"], kind="stable")

    if df.empty:
        raise ValueError(f"No history found for team {team!r}.")

    if latest:
        cutoff = df["date"].max() + pd.Timedelta(nanoseconds=1)
        feature_date = df["date"].max()
    else:
        if end_date is None:
            raise ValueError("end_date is required when latest=False.")
        cutoff = pd.Timestamp(end_date)
        feature_date = cutoff

    history = df.loc[df["date"] < cutoff].copy()
    if start_date is not None:
        history = history.loc[history["date"] >= pd.Timestamp(start_date)]

    if history.empty:
        raise ValueError(
            f"No matches for {team!r} exist in the requested historical window."
        )

    if "points" not in history.columns:
        history["points"] = np.select(
            [
                history["score"] > history["opponent_score"],
                history["score"] == history["opponent_score"],
            ],
            [3, 1],
            default=0,
        )

    score = pd.to_numeric(history["score"], errors="coerce")
    conceded = pd.to_numeric(history["opponent_score"], errors="coerce")
    goal_difference = score - conceded
    total_goals = score + conceded

    ppg_5 = _mean_last(history["points"], 5)
    ppg_10 = _mean_last(history["points"], 10)
    gd_5 = _mean_last(goal_difference, 5)
    gd_10 = _mean_last(goal_difference, 10)

    result: dict[str, Any] = {
        "team": team,
        "date": feature_date,
        "ppg_last_5": ppg_5,
        "ppg_last_10": ppg_10,
        "avg_goals_scored_last_5": _mean_last(score, 5),
        "avg_goals_conceded_last_5": _mean_last(conceded, 5),
        "clean_sheets_rate_last_5": _mean_last(conceded.eq(0).astype(float), 5),
        "avg_goal_difference_last_5": gd_5,
        "avg_goal_difference_last_10": gd_10,
        "btts_rate_last_5": _mean_last(((score > 0) & (conceded > 0)).astype(float), 5),
        "failed_score_rate_last_5": _mean_last(score.eq(0).astype(float), 5),
        "over_2_5_rate_last_5": _mean_last(total_goals.gt(2.5).astype(float), 5),
        "over_3_5_rate_last_5": _mean_last(total_goals.gt(3.5).astype(float), 5),
        "under_1_5_rate_last_5": _mean_last(total_goals.lt(1.5).astype(float), 5),
        "days_since_last_game": max((feature_date - history["date"].max()).days, 0),
        "goal_difference_trend": gd_5 - gd_10,
        "momentum": ppg_5 - ppg_10,
    }

    rating_rows = df.loc[df["date"] <= cutoff].sort_values("date", kind="stable")
    rating_row = rating_rows.iloc[-1]
    for column in _RATING_COLUMNS:
        result[column] = rating_row[column] if column in rating_row.index else np.nan

    return pd.DataFrame([result], columns=TEAM_FEATURE_COLUMNS)


def create_match_features(
    home_team_features: pd.Series | Mapping[str, Any] | pd.DataFrame,
    away_team_features: pd.Series | Mapping[str, Any] | pd.DataFrame,
    *,
    neutral: bool = False,
) -> pd.DataFrame:
    """Create the notebook's model features for one proposed match.

    Parameters
    ----------
    home_team_features, away_team_features:
        Outputs from ``get_team_features`` or equivalent records containing
        the same columns.
    neutral:
        Whether the proposed match is at a neutral venue.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame with prefixed team features plus all engineered
        match-level features from the notebook.
    """
    home = _as_record(home_team_features)
    away = _as_record(away_team_features)

    required = set(TEAM_FEATURE_COLUMNS) - {"date", "pi_neutral_rating"}
    missing_home = sorted(required.difference(home))
    missing_away = sorted(required.difference(away))
    if missing_home or missing_away:
        raise ValueError(
            "Missing required team features. "
            f"home={missing_home or 'none'}, away={missing_away or 'none'}"
        )

    row: dict[str, Any] = {
        "home_team": home["team"],
        "away_team": away["team"],
        "home_neutral": int(neutral),
    }

    prefix_columns = [c for c in TEAM_FEATURE_COLUMNS if c not in {"team", "date"}]
    for column in prefix_columns:
        row[f"home_{column}"] = home.get(column, np.nan)
        row[f"away_{column}"] = away.get(column, np.nan)

    if neutral:
        # This matches the notebook, which compares away ratings on neutral ground.
        row["pi_rating_diff"] = home["pi_away_rating"] - away["pi_away_rating"]
    else:
        row["pi_rating_diff"] = home["pi_home_rating"] - away["pi_away_rating"]

    row["diff_ppg_last_5"] = home["ppg_last_5"] - away["ppg_last_5"]
    row["diff_goal_difference"] = (
        home["avg_goal_difference_last_5"] - away["avg_goal_difference_last_5"]
    )
    row["diff_btts_rate"] = home["btts_rate_last_5"] - away["btts_rate_last_5"]
    row["diff_avg_goals_scored_last_5"] = (
        home["avg_goals_scored_last_5"] - away["avg_goals_scored_last_5"]
    )
    row["diff_avg_goals_conceded_last_5"] = (
        away["avg_goals_conceded_last_5"] - home["avg_goals_conceded_last_5"]
    )
    row["diff_clean_sheets_rate"] = (
        home["clean_sheets_rate_last_5"] - away["clean_sheets_rate_last_5"]
    )
    row["diff_over_2_5_rate"] = (
        home["over_2_5_rate_last_5"] - away["over_2_5_rate_last_5"]
    )
    row["diff_under_1_5_rate"] = (
        home["under_1_5_rate_last_5"] - away["under_1_5_rate_last_5"]
    )
    row["diff_rests_days"] = (
        home["days_since_last_game"] - away["days_since_last_game"]
    )
    row["expected_gd_edge"] = home["pi_expected_gd"] - away["pi_expected_gd"]
    row["diff_momentum"] = home["momentum"] - away["momentum"]

    row["home_pi_momentum"] = home["pi_home_rating"] * home["momentum"]
    row["away_pi_momentum"] = away["pi_away_rating"] * away["momentum"]
    row["home_weighted_avg_ppg"] = 0.7 * home["ppg_last_5"] + 0.3 * home["ppg_last_10"]
    row["away_weighted_avg_ppg"] = 0.7 * away["ppg_last_5"] + 0.3 * away["ppg_last_10"]

    def safe_ratio(numerator: Any, denominator: Any) -> float:
        numerator = float(numerator)
        denominator = float(denominator)
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
            return 0.0
        value = numerator / denominator
        return value if np.isfinite(value) else 0.0

    row["home_goal_to_concede_rate"] = safe_ratio(
        home["avg_goals_scored_last_5"], home["avg_goals_conceded_last_5"]
    )
    row["away_goal_to_concede_rate"] = safe_ratio(
        away["avg_goals_scored_last_5"], away["avg_goals_conceded_last_5"]
    )

    return pd.DataFrame([row])