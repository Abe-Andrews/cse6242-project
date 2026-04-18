import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# classic train test splt
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)

# xgb params
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(labler.classes_),
    eval_metric= 'mlogloss',     
    random_state=42,
    enable_categorical=True, 
    tree_method='hist'      
)

# weight the samples so class dominates the rest
#sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# sample training for tuning
X_tune_A = X_train.sample(n=150000,random_state=42)
pos_idx_A = X_train.index.get_indexer(X_tune_A.index)
y_tune_A = y_train[pos_idx_A]

# weight the samples so class dominates the rest, included just tuning
sample_weights_tune_A = compute_sample_weight(class_weight='balanced', y=y_tune_A)
sample_weights_full = compute_sample_weight(class_weight='balanced', y=y_train)


# randomized search for model a
random_search_A = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=1,
    random_state=42
)


print("Tuning Model A: Pitch Outcome")
random_search_A.fit(X_tune_A,y_tune_A,sample_weight=sample_weights_tune_A)

# best params
print("Best Paramaters for Model A:")
print(random_search_A.best_params_)

# retrain on best model
best_model_A = random_search_A.best_estimator_
best_model_A.fit(X_train,y_train,sample_weight=sample_weights_full)

# prediction
predictions = best_model_A.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, predictions)

# fit the model/make preds
#print("Training model A: Pitch Outcome")
#model.fit(X_train, y_train, sample_weight=sample_weights)
#predictions = model.predict(X_test)

# Get accuracy
#accuracy = accuracy_score(y_test, predictions)

print("\n---------------MODEL A CLASSIFICATION REPORT (Pitching only) ----------------")
print(classification_report(y_test, predictions, target_names=labler.classes_))



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

# initialize Model B
model_B = XGBClassifier(
    objective='multi:softmax',
    num_class=len(labler_B.classes_),
    eval_metric='mlogloss',     
    random_state=42,
    enable_categorical=True, 
    tree_method='hist'       
)

param_dist_B = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# sample training data for tuning
X_tune_B = X_train_B.sample(n=min(100000, len(X_train_B)), random_state=42)
pos_idx_B = X_train_B.index.get_indexer(X_tune_B.index)
y_tune_B = y_train_B[pos_idx_B]

# weights for tuning subset and full training set
sample_weights_tune_B = compute_sample_weight(class_weight='balanced', y=y_tune_B)
sample_weights_full_B = compute_sample_weight(class_weight='balanced', y=y_train_B)

# randomized search
random_search_B = RandomizedSearchCV(
    estimator=model_B,
    param_distributions=param_dist_B,
    n_iter=8,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=1,
    random_state=42
)

print("\nTuning Model B: Batted Outcome")
random_search_B.fit(X_tune_B, y_tune_B, sample_weight=sample_weights_tune_B)

print("Best Parameters for Model B:")
print(random_search_B.best_params_)

# retrain best model on full training data
best_model_B = random_search_B.best_estimator_
best_model_B.fit(X_train_B, y_train_B, sample_weight=sample_weights_full_B)

predictions_B = best_model_B.predict(X_test_B)


print("\n---------------MODEL B CLASSIFICATION REPORT (Assuming the ball was in play) ----------------")
print(classification_report(y_test_B, predictions_B, target_names=labler_B.classes_))


'''
####################### FASTBALLS ONLY ##################
              precision    recall  f1-score   support

        ball       0.82      0.86      0.84     34917
   foul_ball       0.42      0.36      0.39     22939
     in_play       0.37      0.44      0.40     16696
      strike       0.56      0.52      0.54     27105

    accuracy                           0.59    101657
   macro avg       0.54      0.55      0.54    101657
weighted avg       0.59      0.59      0.59    101657

################### PITCH AS CATEGORICAL ##################
              precision    recall  f1-score   support

        ball       0.80      0.84      0.82    107025
   foul_ball       0.39      0.34      0.36     56423
     in_play       0.40      0.50      0.45     52110
      strike       0.57      0.49      0.53     82071

    accuracy                           0.59    297629
   macro avg       0.54      0.54      0.54    297629
weighted avg       0.59      0.59      0.59    297629

################### PITCH AS CATEGORICAL - HYPERPARAMETER ##################
              precision    recall  f1-score   support

        ball       0.80      0.86      0.83    250485       
   foul_ball       0.39      0.33      0.36    134365       
     in_play       0.40      0.51      0.45    122541       
      strike       0.57      0.49      0.52    194017       

    accuracy                           0.59    701408       
   macro avg       0.54      0.55      0.54    701408 

########### CATEGORICAL PITCH AND LESS CLASSES ############
              precision    recall  f1-score   support

        ball       0.79      0.86      0.82    107025
     contact       0.70      0.66      0.68    108533
      strike       0.55      0.54      0.54     82071

    accuracy                           0.70    297629
   macro avg       0.68      0.68      0.68    297629
weighted avg       0.69      0.70      0.69    297629
'''

# Maybe merge foul_ball and in play to one "Contact" target.
"""
---------------MODEL B CLASSIFICATION REPORT (Assuming the ball was in play) ----------------
              precision    recall  f1-score   support

      double       0.67      0.47      0.55      6205
    home_run       0.82      0.89      0.85      4368
         out       0.90      0.95      0.92     65128
      single       0.82      0.74      0.78     20016
      triple       0.39      0.01      0.03       532

    accuracy                           0.87     96249
   macro avg       0.72      0.61      0.63     96249
weighted avg       0.86      0.87      0.86     96249

hyperparameter
---------------MODEL B CLASSIFICATION REPORT (Assuming the ball was in play) ----------------
              precision    recall  f1-score   support

      double       0.47      0.49      0.48      7813       
    home_run       0.77      0.90      0.83      5534       
         out       0.94      0.83      0.88     82679       
      single       0.75      0.74      0.75     25434       
      triple       0.04      0.61      0.08       663       

    accuracy                           0.79    122123       
   macro avg       0.60      0.71      0.60    122123       
weighted avg       0.86      0.79      0.82    122123 
"""