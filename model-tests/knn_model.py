import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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

# classic train test splt
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)


numeric_features = [
    'plate_x',
    'plate_z',
    'release_speed',
    'pfx_x',
    'pfx_z',
    'balls',
    'strikes',
    'stand',
    'p_throws',
    'release_extension',
    'arm_angle'
]

categorical_features = ['pitch_name']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

knn_model = Pipeline([
    ('prep', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))
])

param_grid_A = {
    'knn__n_neighbors': [5, 15, 25],
    'knn__weights': ['uniform', 'distance']
}

X_tune_knn_A = X_train.sample(n=100000, random_state=42)
pos_idx_A = X_train.index.get_indexer(X_tune_knn_A.index)
y_tune_knn_A = y_train[pos_idx_A]

grid_knn_A = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid_A,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=1
)

print("Tuning KNN Model A")
grid_knn_A.fit(X_tune_knn_A, y_tune_knn_A)

print("Best KNN Parameters for Model A:")
print(grid_knn_A.best_params_)

# retrain best model on full data
print("Training KNN Model A")
best_knn_A = grid_knn_A.best_estimator_
best_knn_A.fit(X_train, y_train)

knn_preds = best_knn_A.predict(X_test)

print("\n---------------KNN MODEL A CLASSIFICATION REPORT----------------")
print(classification_report(y_test, knn_preds, target_names=labler.classes_))

#---------------KNN MODEL A CLASSIFICATION REPORT----------------
#              precision    recall  f1-score   support
#
#        ball       0.78      0.84      0.80    250485
#   foul_ball       0.37      0.32      0.34    134365
#     in_play       0.39      0.35      0.37    122541
#      strike       0.50      0.54      0.52    194017
#
#    accuracy                           0.57    701408
#   macro avg       0.51      0.51      0.51    701408
#weighted avg       0.56      0.57      0.56    701408

# hyperparameter
#---------------KNN MODEL A CLASSIFICATION REPORT----------------
#              precision    recall  f1-score   support
#
#        ball       0.77      0.85      0.81    250485       
#   foul_ball       0.38      0.34      0.36    134365       
#     in_play       0.41      0.35      0.37    122541       
#      strike       0.52      0.53      0.53    194017       
#
#    accuracy                           0.58    701408       
#   macro avg       0.52      0.52      0.52    701408       
#weighted avg       0.56      0.58      0.57    701408       


############################## KNN MODEL B (Post contact outcome) #####################################

contact_features = features + [
    'launch_speed_angle',
    'hc_x',
    'hc_y',
    'bb_type'
]

contact_target = 'hit_result'

# in play only
df_B = df[df['description'] == 'hit_into_play'].copy()

# hit types
hit_types = ['single', 'double', 'triple', 'home_run']

# label all non-hit balls in play as out
df_B['hit_result'] = np.where(df_B['events'].isin(hit_types), df_B['events'], 'out')

# create dataset B
df_B = df_B[contact_features + [contact_target]].dropna()

# split features / target
X_B = df_B[contact_features]
y_B = df_B[contact_target]

# encode target
labler_B = LabelEncoder()
y_B_labeled = labler_B.fit_transform(y_B)

# split
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B_labeled, test_size=0.2, random_state=42, stratify=y_B_labeled
)

# numeric / categorical features for model B
numeric_features_B = [
    'plate_x',
    'plate_z',
    'release_speed',
    'pfx_x',
    'pfx_z',
    'balls',
    'strikes',
    'stand',
    'p_throws',
    'release_extension',
    'arm_angle',
    'launch_speed_angle',
    'hc_x',
    'hc_y'
]

categorical_features_B = ['pitch_name', 'bb_type']

preprocessor_B = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_B),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_B)
    ]
)

knn_model_B = Pipeline([
    ('prep', preprocessor_B),
    ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))
])

param_grid_B = {
    'knn__n_neighbors': [5, 15, 25],
    'knn__weights': ['uniform', 'distance']
}

# sample for tuning
X_tune_knn_B = X_train_B.sample(n=min(80000, len(X_train_B)), random_state=42)
pos_idx_B = X_train_B.index.get_indexer(X_tune_knn_B.index)
y_tune_knn_B = y_train_B[pos_idx_B]

grid_knn_B = GridSearchCV(
    estimator=knn_model_B,
    param_grid=param_grid_B,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=1
)

print("\nTuning KNN Model B")
grid_knn_B.fit(X_tune_knn_B, y_tune_knn_B)

print("Best KNN Parameters for Model B:")
print(grid_knn_B.best_params_)

# retrain best model on full data
print("Training Model B")
best_knn_B = grid_knn_B.best_estimator_
best_knn_B.fit(X_train_B, y_train_B)

knn_preds_B = best_knn_B.predict(X_test_B)

print("\n---------------KNN MODEL B CLASSIFICATION REPORT----------------")
print(classification_report(y_test_B, knn_preds_B, target_names=labler_B.classes_))

#---------------KNN MODEL B CLASSIFICATION REPORT----------------
#              precision    recall  f1-score   support
#
#      double       0.60      0.23      0.33      7813
#    home_run       0.74      0.70      0.72      5534
#         out       0.85      0.93      0.89     82679
#      single       0.67      0.63      0.65     25434
#      triple       0.00      0.00      0.00       663

#    accuracy                           0.81    122123
#   macro avg       0.57      0.50      0.52    122123
#weighted avg       0.79      0.81      0.79    122123

# hyperparameter
#---------------KNN MODEL B CLASSIFICATION REPORT----------------
#              precision    recall  f1-score   support
#
#      double       0.60      0.23      0.33      7813
#    home_run       0.74      0.70      0.72      5534
#         out       0.85      0.93      0.89     82679
#      single       0.67      0.63      0.65     25434
#      triple       0.00      0.00      0.00       663
#
#    accuracy                           0.81    122123
#   macro avg       0.57      0.50      0.52    122123
#weighted avg       0.79      0.81      0.79    122123