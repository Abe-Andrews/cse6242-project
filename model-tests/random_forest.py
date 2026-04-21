import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
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
    'release_spin_rate',
    'pfx_x', # Horizontal pitch movement
    'pfx_z', # Vertical pitch movement
    'balls', # Current number of balls
    'strikes', # Current number of strikes
    'stand', # Batter handedness
    'p_throws', # Pitcher handedness
    'release_extension',
    'arm_angle',
    'outs_when_up'
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

# classic train test splt
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)

# have to convert categorical features to numeric
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
X_train_encoded['pitch_name'] = X_train_encoded['pitch_name'].cat.codes
X_test_encoded['pitch_name'] = X_test_encoded['pitch_name'].cat.codes

# random forest params
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# fit the model/make preds
print("Training model A: Pitch Outcome")
model.fit(X_train_encoded, y_train)
predictions = model.predict(X_test_encoded)

# Get accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n---------------MODEL A CLASSIFICATION REPORT (Pitching only) ----------------")
print(classification_report(y_test, predictions, target_names=labler.classes_, zero_division=0))

joblib.dump(model, 'models/rf_pitch_outcome_model.joblib')
joblib.dump(labler, 'models/rf_pitch_outcome_labeler.joblib')
print("\nModel A saved to models/rf_pitch_outcome_model.joblib")



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

# train test split
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B_labeled, test_size=0.2, random_state=42, stratify=y_B_labeled
)

X_train_B_encoded = X_train_B.copy()
X_test_B_encoded = X_test_B.copy()
X_train_B_encoded['pitch_name'] = X_train_B_encoded['pitch_name'].cat.codes
X_test_B_encoded['pitch_name'] = X_test_B_encoded['pitch_name'].cat.codes
X_train_B_encoded['bb_type'] = X_train_B_encoded['bb_type'].cat.codes
X_test_B_encoded['bb_type'] = X_test_B_encoded['bb_type'].cat.codes

# initialize Model B
model_B = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# fit Model B
print("\nTraining model B: Batted Outcome")
model_B.fit(X_train_B_encoded, y_train_B)
predictions_B = model_B.predict(X_test_B_encoded)

print("\n---------------MODEL B CLASSIFICATION REPORT (Assuming the ball was in play) ----------------")
print(classification_report(y_test_B, predictions_B, target_names=labler_B.classes_, zero_division=0))

joblib.dump(model_B, 'models/rf_batted_outcome_model.joblib')
joblib.dump(labler_B, 'models/rf_batted_outcome_labeler.joblib')
print("\nModel B saved to models/rf_batted_outcome_model.joblib")


"""
Training model A: Pitch Outcome

---------------MODEL A CLASSIFICATION REPORT (Pitching only) ----------------
              precision    recall  f1-score   support

        ball       0.76      0.89      0.82    250485
   foul_ball       0.40      0.31      0.35    134365
     in_play       0.43      0.34      0.38    122541
      strike       0.53      0.57      0.55    194017

    accuracy                           0.60    701408
   macro avg       0.53      0.53      0.53    701408
weighted avg       0.57      0.60      0.58    701408


Training model B: Batted Outcome

---------------MODEL B CLASSIFICATION REPORT (Assuming the ball was in play) ----------------
              precision    recall  f1-score   support

      double       0.71      0.53      0.61      7813
    home_run       0.85      0.84      0.85      5534
         out       0.90      0.96      0.93     82679
      single       0.82      0.74      0.78     25434
      triple       0.00      0.00      0.00       663

    accuracy                           0.87    122123
   macro avg       0.65      0.61      0.63    122123
weighted avg       0.86      0.87      0.87    122123
"""

