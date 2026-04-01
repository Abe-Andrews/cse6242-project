import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


# load data
df = pd.read_csv("mlb_dataset.csv")


# Make the pitch_name a categorical type
df['pitch_name'] = df['pitch_name'].astype('category')

# just one in_play class
df['event_group'] = df['event_group'].replace({'inplay_hit': 'in_play','inplay_out': 'in_play'})

# Replace foul_ball and in_play to contact
# df['event_group'] = df['event_group'].replace({'foul_ball': 'contact', 'in_play': 'contact'})

# set features here
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
    'p_throws' # Pitcher handedness

    # BATTING METRICS

    # TO BE ADDED FROM CSV
    '''
    'launch_speed_angle'
    'release_extention',
    'arm_angle'
    '''
]

# dependent variable
target = 'event_group'

# encode these
df['stand'] = df['stand'].map({'L': 0, 'R': 1})
df['p_throws'] = df['p_throws'].map({'L': 0, 'R': 1})

# create dataset
df = df[features + [target]].dropna()

# split df
X = df[features]
y = df[target]

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
    enable_categorical=True, # THIS IS THE MAGIC BULLET
    tree_method='hist'       # Highly recommended when using categorical data in XGBoost

)

# weight the samples so class dominates the rest
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# fit the model/make preds
model.fit(X_train, y_train, sample_weight=sample_weights)
predictions = model.predict(X_test)

# Get accuracy
accuracy = accuracy_score(y_test, predictions)

print(classification_report(y_test, predictions, target_names=labler.classes_))
#print(df.head())

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