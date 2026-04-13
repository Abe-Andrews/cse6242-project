import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


# load data
#df = pd.read_csv("mlb_dataset.csv")
df = pd.read_csv("data/data.csv")

strike_mask = df['description'].isin(['called_strike','swinging_strike','swinging_strike_blocked','missed_bunt'])
ball_mask = df['description'].isin(['ball','blocked_ball','intent_ball','pitchout','automatic_ball'])
foul_mask = df['description'].isin(['foul','foul_tip','foul_bunt','bunt_foul_tip'])
inplay_hit_mask = ((df['description'] == 'hit_into_play') &(df['events'].isin(['single', 'double', 'triple', 'home_run'])))
inplay_out_mask = ((df['description'] == 'hit_into_play') &(~df['events'].isin(['single', 'double', 'triple', 'home_run'])))

df['event_group'] = np.select(
    [strike_mask, ball_mask, foul_mask, inplay_hit_mask, inplay_out_mask],
    ['strike', 'ball', 'foul_ball', 'inplay_hit', 'inplay_out'],
    default='other'
)
# Make the pitch_name a categorical type
df['pitch_name'] = df['pitch_name'].astype('category')
df['bb_type'] = df['bb_type'].astype('category')


# just one in_play class
df['event_group'] = df['event_group'].replace({'inplay_hit': 'in_play','inplay_out': 'in_play'})

# encode these
df['stand'] = df['stand'].map({'L': 0, 'R': 1})
df['p_throws'] = df['p_throws'].map({'L': 0, 'R': 1})

# set features here
# MODEL A
features = [
    # PITCHING METRICS
    'pitch_name', 
    'plate_x', # Horizontal pitch location
    'plate_z', # Vertival pitch location
    'release_speed', # Release Speed of Pitch
    'pfx_x', # Horizontal pitch movement
    'pfx_z', # Vertical pitch movement
    'balls', # Current number of balls
    'strikes', # Current number of strikes
    'stand', # Batter handedness
    'p_throws', # Pitcher handedness
    'release_extension',
    'arm_angle'
  ]

# dependent variable
target = 'event_group'

# create dataset
df_a = df[features + [target]].dropna()
df_a = df_a[df_a['event_group'] != 'other']

# split df
X = df_a[features]
y = df_a[target]

# this allows us to number each class ('event_group')
labler = LabelEncoder()
y_labeled = labler.fit_transform(y)

# xgb params
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(labler.classes_),
    eval_metric= 'mlogloss',     
    random_state=42,
    enable_categorical=True, 
    tree_method='hist'      

)

# weight the samples so class dominates the rest
sample_weights = compute_sample_weight(class_weight='balanced', y=y_labeled)

# fit the model/make preds
print("Training model A: Pitch Outcome")
model.fit(X, y_labeled, sample_weight=sample_weights)

# Save A
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model,'models/pitch_outcome_model.joblib')
joblib.dump(labler, 'models/pitch_outcome_labeler.joblib')

print("\n------------MODEL A SAVED------------")



############################## MODEL 2 (Post contact outcome) #####################################
contact_features = features + ['launch_speed_angle',
                               'hc_x',
                               'hc_y',
                               'bb_type' # batted ball type (pop up, line drive, etc)
                               ]

contact_target = 'hit_result'

# In play only
df_B = df[df['description'] == 'hit_into_play'].copy()

# Hits to predict (ball in)
hit_types = ['single', 'double', 'triple', 'home_run']

# If the event is a hit type, keep its name. If not, label it 'out'
df_B['hit_result'] = np.where(df_B['events'].isin(hit_types), df_B['events'], 'out')

# create dataset B
df_B = df_B[contact_features + [contact_target]].dropna()

# split df_B
X_B = df_B[contact_features]
y_B = df_B[contact_target]

# label the new target 
labler_B = LabelEncoder()
y_B_labeled = labler_B.fit_transform(y_B)

# initialize Model B
model_B = XGBClassifier(
    objective='multi:softprob',
    num_class=len(labler_B.classes_),
    eval_metric='mlogloss',     
    random_state=42,
    enable_categorical=True, 
    tree_method='hist'       
)

# fit Model B
print("\nTraining model B: Batted Outcome")
model_B.fit(X_B, y_B_labeled)

joblib.dump(model_B, 'models/batted_outcome_model.joblib')
joblib.dump(labler_B, 'models/batted_outcome_labler.joblib')
print("\n------------MODEL B SAVED------------")

