import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('dataframe_hrv.csv', sep=',', dtype=np.float64)
def missing_values(df):
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
    df.plot( y=["HR"])
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13) 
    df.plot( y=["HR"])
    df=df.fillna(df.mean(),inplace=False)
    return df

def fix_stress_labels(df='',label_column='stress'):
    df['stress'] = np.where(df['stress']>=0.5, 1, 0)
    display(df["stress"].unique())
    return df

tpot_data = fix_stress_labels(df=tpot_data)
tpot_data = missing_values(tpot_data)

selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']

X = tpot_data[selected_x_columns]
y = tpot_data['stress']

features = tpot_data.drop(selected_x_columns, axis=1).values


training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['stress'].values, random_state=None)

# Average CV score on the training set was:0.7826198179043885
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            VarianceThreshold(threshold=0.2),
            RBFSampler(gamma=0.15000000000000002)
        ),
        FunctionTransformer(copy)
    ),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.55, min_samples_leaf=4, min_samples_split=11, n_estimators=100, subsample=0.9000000000000001)
)



exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
