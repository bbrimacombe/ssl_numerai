import time
start = time.time()

import pandas as pd
from lightgbm import LGBMRegressor
import gc
import json

from numerapi import NumerAPI
from halo import Halo
from utils import (
    save_model,
    load_model,
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL
)



napi = NumerAPI()
spinner = Halo(text='', spinner='dots')

current_round = 305 #napi.get_current_round()

# Tournament data changes every week so we specify the round in their name. Training
# and validation data only change periodically, so no need to download them every time.
print('Downloading dataset files...')
#napi.download_dataset("numerai_training_data.parquet", "training_data.parquet")
#napi.download_dataset("numerai_tournament_data.parquet", f"tournament_data_{current_round}.parquet")
#napi.download_dataset("numerai_validation_data.parquet", f"validation_data.parquet")
#napi.download_dataset("example_validation_predictions.parquet")
#napi.download_dataset("features.json")


"""
print('Reading minimal training data')
# read the feature metadata and get the "small" feature set
with open("features.json", "r") as f:
    feature_metadata = json.load(f)
"""

#############
import numpy as np
synth = True 
use_era = False
synth_type = np.float16

model_name = "model_target_synth5_02" 
synth_data = ["synth_feat6_training.pkl"] 
synth_data_val = ["synth_feat6_validation.pkl"] 
synth_data_tourn = ["synth_feat6_tournament.pkl"]


#
def normalize(data):
    ###Argmax of the features
    features = [c for c in data if c.startswith("feature_")]

    #data["feature_era_infer_max"] = np.argmax(data.to_numpy(), axis=-1) / 12 #np.argmax(data.to_numpy(), axis=-1) / 7
    data["feature_era_infer"] = np.argmax(data[features].to_numpy(), axis=-1) / 12

    #Didn't help at all
    #data[features] = data.groupby(ERA_COL).transform(lambda x: (x - x.mean()) / x.std())
    #Didn't help at all
    #data[["feature_top_1", "feature_top_2"]] = data[features].to_numpy().argsort(axis=-1)[:, -2:] / 12  #[::-1] 9, 1, 11

    print("----------------------")
    #print(data.groupby(ERA_COL).std())
    data = data.drop(ERA_COL, axis=1)
    data = data.drop(features, axis=1)

    return data

 

training_data = pd.read_parquet('training_data.parquet')


#############
#features = feature_metadata["feature_sets"]["small"]
features = [c for c in training_data if c.startswith("feature_")]

read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
###########

#Try only using ssl features
#################################
bins = []

if synth:

    synth_train = pd.read_pickle(synth_data[0])
    #synth_train[ERA_COL] = training_data[ERA_COL] #Optional

    synth_train = normalize(synth_train)
    for i in range(1, len(synth_data)):
        synth_train += normalize(pd.read_pickle(synth_data[i]))



    synth_features = [c for c in synth_train if c.startswith("feature_")]

    print(synth_train)

    original_features = features.copy()
    features += synth_features
    #features = synth_features

    #print(synth_train)

    training_data = pd.concat([training_data, synth_train], axis=1)

#################################

if use_era:
    era_vals = training_data["era"].unique()
    max_era = era_vals.max()
    word2id = {}
    for idx, name in enumerate(era_vals):
        word2id[name] = idx
    era_idx = training_data["era"].map(word2id)
    era_idx = np.interp(era_idx, [0, max_era], [0, 11]) #Interpolate to smaller scale
    era_idx = np.rint(era_idx).astype(np.int16) #Round down to nearest
    training_data["feature_era"] = era_idx / 11
    #training_data["feature_era_infer"] += era_idx / 11
    #features += ["feature_era"]

    print(training_data)
    #print(training_data["feature_era_infer"])






# getting the per era correlation of each feature vs the target
all_feature_corrs = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)


# find the riskiest features by comparing their correlation vs
# the target in each half of training data; we'll use these later
riskiest_features = get_biggest_change_features(all_feature_corrs, 0) #50
print("Riskiest features are:")
print(riskiest_features)


# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()


print(f"Checking for existing model '{model_name}'")
model = load_model(model_name)


if not model:
    print(f"model not found, creating new one")

    #"""
    params = {"n_estimators": 2000, #2000 , #1500 now for ram puprose
              "learning_rate": 0.01,
              "max_depth": 5, #5
              "num_leaves": 2 ** 5,
              "colsample_bytree": 0.1}

    model = LGBMRegressor(**params)
    #"""
    #from sklearn.neural_network import MLPRegressor
   # model = MLPRegressor(hidden_layer_sizes=(100,))



    # train on all of train and save the model so we don't have to train next time
    spinner.start('Training model')
    model.fit(training_data.filter(like='feature_', axis='columns'),
              training_data[TARGET_COL])
    print(f"saving new model: {model_name}")
    save_model(model, model_name)
    spinner.succeed()

gc.collect()

print('Reading minimal features of validation and tournament data...')
validation_data = pd.read_parquet('numerai_validation_data.parquet',
                                  columns=read_columns)
tournament_data = pd.read_parquet(f'numerai_tournament_data_{current_round}.parquet',
                                  columns=read_columns)

###################

if synth:
    synth_val = pd.read_pickle(synth_data_val[0])
    #synth_val[ERA_COL] = validation_data[ERA_COL] #Optional

    synth_val = normalize(synth_val)

    for i in range(1, len(synth_data_val)):
        synth_val += normalize(pd.read_pickle(synth_data_val[i]))


    validation_data = pd.concat([validation_data, synth_val], axis=1)
    tournament_data = pd.concat([tournament_data, synth_val[:len(tournament_data)]], axis=1)

    #For synth only
    #validation_data = validation_data.drop(original_features)
    #tournament_data = tournament_data.drop(original_features)
    
    print("Remaining columns...")
    print(validation_data.columns)

####################
if use_era:
    validation_data["feature_era"] = 1.0
    tournament_data["feature_era"] = 1.0

    #validation_data["feature_era_infer"] += 1.0
    #tournament_data["feature_era_infer"] += 1.0
####################



nans_per_col = tournament_data[tournament_data["data_type"] == "live"].isna().sum()

# check for nans and fill nans
if nans_per_col.any():
    total_rows = len(tournament_data[tournament_data["data_type"] == "live"])
    print(f"Number of nans per column this week: {nans_per_col[nans_per_col > 0]}")
    print(f"out of {total_rows} total rows")
    print(f"filling nans with 0.5")
    tournament_data.loc[:, features] = tournament_data.loc[:, features].fillna(0.5)
else:
    print("No nans in the features this week!")


spinner.start('Predicting on validation and tournament data')
# double check the feature that the model expects vs what is available to prevent our
# pipeline from failing if Numerai adds more data and we don't have time to retrain!
model_expected_features = model.booster_.feature_name()
if set(model_expected_features) != set(features):
    print(f"New features are available! Might want to retrain model {model_name}.")

#print(validation_data.loc[:, model_expected_features])
"""
print("\nAbout to predict on this:")
print(model_expected_features)
print("~~~~")
print(validation_data[model_expected_features])
print("~~+++~~")
print(validation_data.isna().sum())
print("----------------")
print(validation_data.loc[:, model_expected_features])
print("-------.............................................")
"""

validation_data.loc[:, f"preds_{model_name}"] = model.predict(validation_data.loc[:, model_expected_features])


tournament_data.loc[:, f"preds_{model_name}"] = model.predict(tournament_data.loc[:, model_expected_features])
spinner.succeed()

gc.collect()

spinner.start('Neutralizing to risky features')

# neutralize our predictions to the riskiest features
validation_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
    df=validation_data,
    columns=[f"preds_{model_name}"],
    neutralizers=riskiest_features,
    proportion=1.0,
    normalize=True,
    era_col=ERA_COL
)

"""
tournament_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
    df=tournament_data,
    columns=[f"preds_{model_name}"],
    neutralizers=riskiest_features,
    proportion=1.0,
    normalize=True,
    era_col=ERA_COL
)
"""
spinner.succeed()


model_to_submit = f"preds_{model_name}_neutral_riskiest_50"

# rename best model to "prediction" and rank from 0 to 1 to meet upload requirements
validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)
#tournament_data["prediction"] = tournament_data[model_to_submit].rank(pct=True)
validation_data["prediction"].to_csv(f"validation_predictions_{current_round}.csv")
#tournament_data["prediction"].to_csv(f"tournament_predictions_{current_round}.csv")

spinner.start('Reading example validation predictions')
validation_preds = pd.read_parquet('example_validation_predictions.parquet')
validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]
spinner.succeed()

# get some stats about each of our models to compare...
# fast_mode=True so that we skip some of the stats that are slower to calculate
validation_stats = validation_metrics(validation_data, [model_to_submit], example_col=EXAMPLE_PREDS_COL, fast_mode=True)
#print(validation_stats[["mean", "sharpe"]].to_markdown())
print(validation_stats.to_markdown())


#from utils import save_corr_csv
#save_corr_csv(validation_data, [model_to_submit], path="random_model_01.csv")  

#Save example preds
#validation_data[model_to_submit] = validation_data[EXAMPLE_PREDS_COL]
#save_corr_csv(validation_data, [model_to_submit], path="example_preds.csv")

"""
print(f'''
Done! Next steps:
    1. Go to numer.ai/tournament (make sure you have an account)
    2. Submit validation_predictions_{current_round}.csv to the diagnostics tool
    3. Submit tournament_predictions_{current_round}.csv to the "Upload Predictions" button
''')
"""

print(f'done in {(time.time() - start) / 60} mins')
