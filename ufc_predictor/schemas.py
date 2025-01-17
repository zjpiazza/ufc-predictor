from typing import Dict
import pandas as pd

fights_dtypes: Dict[str, type | pd.core.arrays.integer.Int64Dtype] = {
        "fight_id": str,
        "event_id": str,
        "referee": str,
        "fighter_1": str,
        "fighter_2": str,
        "winner": str,
        "num_rounds": pd.Int64Dtype(),
        "title_fight": str,
        "weight_class": str,
        "gender": str,
        "result": str,
        "result_details": str,
        "finish_round": pd.Int64Dtype(),
        "finish_time": str,
        "time_format": str,
        "scores_1": pd.Int64Dtype(),
        "scores_2": pd.Int64Dtype(),
    }

rounds_dtypes: Dict[str, type | pd.core.arrays.integer.Int64Dtype] = {
    "fight_id": str,
    "fighter_id": str,
    "round": pd.Int64Dtype(),
    "knockdowns": pd.Int64Dtype(),
    "strikes_att": pd.Int64Dtype(),  # If not stated otherwise they are significant
    "strikes_succ": pd.Int64Dtype(),
    "head_strikes_att": pd.Int64Dtype(),
    "head_strikes_succ": pd.Int64Dtype(),
    "body_strikes_att": pd.Int64Dtype(),
    "body_strikes_succ": pd.Int64Dtype(),
    "leg_strikes_att": pd.Int64Dtype(),
    "leg_strikes_succ": pd.Int64Dtype(),
    "distance_strikes_att": pd.Int64Dtype(),
    "distance_strikes_succ": pd.Int64Dtype(),
    "ground_strikes_att": pd.Int64Dtype(),
    "ground_strikes_succ": pd.Int64Dtype(),
    "clinch_strikes_att": pd.Int64Dtype(),
    "clinch_strikes_succ": pd.Int64Dtype(),
    "total_strikes_att": pd.Int64Dtype(),  # significant and not significant
    "total_strikes_succ": pd.Int64Dtype(),
    "takedown_att": pd.Int64Dtype(),
    "takedown_succ": pd.Int64Dtype(),
    "submission_att": pd.Int64Dtype(),
    "reversals": pd.Int64Dtype(),
    "ctrl_time": str,
}


fighters_dtypes: Dict[str, type | pd.core.arrays.integer.Int64Dtype] = {
    "fighter_id": str,
    "fighter_f_name": str,
    "fighter_l_name": str,
    "fighter_nickname": str,
    "fighter_height_cm": float,
    "fighter_weight_lbs": float,
    "fighter_reach_cm": float,
    "fighter_stance": str,
    "fighter_dob": "datetime64[ns]",
    "fighter_w": pd.Int64Dtype(),
    "fighter_l": pd.Int64Dtype(),
    "fighter_d": pd.Int64Dtype(),
    "fighter_nc_dq": pd.Int64Dtype(),
}

events_dtypes: Dict[str, type | pd.core.arrays.integer.Int64Dtype] = {
    "event_id": str,
    "event_name": str,
    "event_date": "datetime64[ns]",
    "event_city": str,
    "event_state": str,
    "event_country": str,
}