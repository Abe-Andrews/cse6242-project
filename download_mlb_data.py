"""
# Before running this script, run the below:

python -m venv .venv
source .venv/bin/activate (on Windows, .venv\Scripts\activate)
pip install -r requirements.txt
"""

from pybaseball import pybaseball, statcast
from utils.cleanup import keep_regular_season_games, normalize_handedness
from tqdm import tqdm
import pandas as pd
import os

pybaseball.cache.enable()
YEARS = [2021, 2022, 2023, 2024, 2025]
COLUMNS_TO_KEEP = [
  # pitch characteristics
  'pitch_type',
  'pitch_name',
  'release_speed',
  'release_pos_x',  # NOTE: (see cleanup utils)
  'release_pos_z',
  'pfx_x',  # NOTE: (see cleanup utils)
  'pfx_z',
  'api_break_z_with_gravity',
  'plate_x',    # NOTE: (see cleanup utils)
  'plate_z',
  'release_spin_rate',
  'release_extension',
  'spin_axis',  # NOTE: (see cleanup utils)
  'arm_angle',  # NOTE: (see cleanup utils)
  
  # swing, contact, and batted ball characteristics
  'launch_speed',
  'launch_angle',
  'bat_speed',
  'swing_length',
  'attack_angle',
  'attack_direction',
  'swing_path_tilt',
  
  # outcome measurements
  'event', 
  'description',
  'des',
  'hit_location',
  'delta_run_exp',
  'woba_value',
  
  # ======================= CONTEXT VARIABLES =======================
  # matchup context
  'batter',
  'pitcher',
  'stand',
  'p_throws',
  'balls',
  'strikes',
  'effective_speed',  # technically misc.
  
  # game context
  'at_bat_number',
  'pitch_number',
  'home_score',
  'away_score',
  'game_type',  # NOTE: (see cleanup utils)
  'game_date',
  'home_team',
  'away_team',
  'game_year',
  'on_3b',
  'on_2b',
  'on_1b',
  'outs_when_up',
  'inning',
  'inning_topbot',
  'n_priorpa_thisgame_player_at_bat',
  'pitcher_days_since_prev_game',
  'batter_days_since_prev_game',
  'n_thruorder_pitcher',
]

# function to retrieve info from statcast
def _pull_year(year: int, start="03-01", end="11-30"):
    start_dt = f"{year}-{start}"
    end_dt   = f"{year}-{end}"
    df = statcast(start_dt, end_dt)
    return df

def save_to_file(df_all):
    # make data folder if not exists
    if not os.path.exists("data"):
        os.makedirs("data")

    df_all.to_csv(f"data/data.csv", index=False)
    print(f"\nSaved to: data/data.csv")
    
def cleanup(df):
    df = keep_regular_season_games(df)
    df = normalize_handedness(df)
    return df

def download():
    # concat dataframes
    dfs = []

    # loop over years
    for y in tqdm(YEARS):
        df_y = _pull_year(y)
        dfs.append(df_y)

    # concat data
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows downloaded: {len(df_all):,}")

    # filter to desired cols
    print(f"Original columns: {len(df_all.columns)}")
    existing_cols = [col for col in COLUMNS_TO_KEEP if col in df_all.columns]

    df_all = df_all[existing_cols]
    print(f"Filtered to {len(df_all.columns)} columns")

    # info about data
    print("\n" + "="*60)
    print(df_all.info())
    
    return df_all

def main():
    df = download()
    df = cleanup(df)
    save_to_file(df)
    return 
    
if __name__ == "__main__":
    main()