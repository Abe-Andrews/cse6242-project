import pandas as pd

def keep_regular_season_games(df):
    """
    Discard all non-regular season games (pre-season, exhibitions, and playoffs).
    """
    return df[df['game_type'] == 'R'].copy()

def normalize_handedness(df):
    """
    We want to flip all the relevant features for left-handed pitchers so that they are in the same frame-of-reference as right-handed pitchers.
    """
    df = df.copy()
    lhp = df['p_throws'] == 'L'
    
    # flip `pfx_x` (horizontal pitch movement) for LHP
    df.loc[lhp, 'pfx_x'] = -df.loc[lhp, 'pfx_x']
    
    # flip `release_pos_x` (horizontal release position) for LHP
    df.loc[lhp, 'release_pos_x'] = -df.loc[lhp, 'release_pos_x']
    
    # flip `spin_axis` around 180 degrees for LHP
    # new = (360 - old) % 360
    df.loc[lhp, 'spin_axis'] = (360 - df.loc[lhp, 'spin_axis']) % 360
    
    # flip `plate_x` (horizontal pitch location) for LHP
    # critically, this  makes location relative to pitcher's perspective
    df.loc[lhp, 'plate_x'] = -df.loc[lhp, 'plate_x']
        
    # mirror `arm_angle` around 180 degrees for LHP
    df.loc[lhp, 'arm_angle'] = 180 - df.loc[lhp, 'arm_angle']
    
    return df
