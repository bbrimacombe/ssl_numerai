import os 
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

def build_model(config):
	#construct keras model graph 

	#"""
	activation = keras.activations.sigmoid #keras.activations.swish #keras.activations.relu #keras.activations.swish
	kernel_initializer = keras.initializers.RandomNormal(stddev=0.05)
	bias_initializer = keras.initializers.RandomNormal(stddev=0.05)


	input_1 = layers.Input((config.NUM_FEATURES), dtype=tf.int8) 
	input_1_cast = layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32))(input_1)
	input_2 = layers.Input((1,), dtype=tf.float32)

	#Training = true for less functional drift during inference
	input_1_cast = layers.Dropout(0.5)(input_1_cast, training=True)
	input_1_cast = layers.Concatenate(axis=-1)([input_1_cast, input_2])

	#training=True bc we want test set information to be same
	itermediate1 = layers.Dense(1051, activation="sigmoid", name="dense0")(input_1_cast)
	itermediate1 = layers.Dropout(0.2)(itermediate1, training=True)
	itermediate1 = layers.Dense(1051, activation="sigmoid", name="dense1")(itermediate1)
	itermediate1 = layers.Dropout(0.2)(itermediate1, training=True)

	itermediate2 = layers.Dense(1051, activation="sigmoid", name="dense3")(itermediate1 + input_1_cast)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)
	itermediate2 = layers.Dense(1051, activation="sigmoid", name="dense4")(itermediate2)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)

	itermediate2 = layers.Dense(1051, activation="sigmoid", name="dense5")(itermediate2 + input_1_cast)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)
	itermediate2 = layers.Dense(1051, activation="sigmoid", name="dense6")(itermediate2)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)

	itermediate2 = layers.Dense(1051, activation="sigmoid", name="dense7")(itermediate2 + input_1_cast)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)
	itermediate2 = layers.Dense(1051, activation="sigmoid", name="dense8")(itermediate2)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)
	

	targets = layers.Dense(config.NUM_TARGETS, activation="sigmoid", name="targets")(itermediate2) #Num eras

	features = layers.Dense(config.NUM_FEATURES, activation="sigmoid")(itermediate2)
	#2x memory efficiency in int targets
	features = layers.Lambda(lambda x: 4*x, name="features")(features) #4*decoder(enc_out) #To span range (integer targets)
	#final = layers.Layer(name="model_out")(final)

	#model = tf.keras.Model([input_1, input_2, input_3, input_4, input_5], [targets, features], name="model")
	model = tf.keras.Model([input_1, input_2], [features, targets], name="model")
	#model = tf.keras.Model([input_1], [targets], name="model")
	#model = tf.keras.Model([input_1, input_2, input_3, input_4, input_5], [targets], name="model")

	print("Built model:")
	model.summary()

	return model

def load_model(config):
	model = tf.keras.models.load_model(config.MODELS_PATH+config.MODEL_NAME) #+".h5"
	model.summary()

	return model


def build_deterministic_partitions(config, data):
	#Feature partitions not implemented in this version
	return (4*data).astype(np.int16)


def build_tf_train(config, x, y, eras):
	feat = (4*x).astype(np.int16)

	print("Spliting train and test (skleran)...")
	from sklearn.model_selection import train_test_split
	data_train, data_test, era_train, era_test, y_train, y_test = train_test_split(feat, eras, y, test_size=0.1, random_state=0) #0.05

	#parr_inputs = tf.data.Dataset.from_tensor_slices((build_deterministic_partitions(config, x)))
	parr_inputs = tf.data.Dataset.from_tensor_slices((data_train, era_train))
	parr_targets = tf.data.Dataset.from_tensor_slices((data_train, y_train)) 
	tf_data = tf.data.Dataset.zip((parr_inputs, parr_targets))
	tf_data = tf_data.shuffle(10000000).batch(config.BATCH_SIZE)

	parr_inputs = tf.data.Dataset.from_tensor_slices((data_test, era_test))
	parr_targets = tf.data.Dataset.from_tensor_slices((data_test, y_test)) 
	tf_test = tf.data.Dataset.zip((parr_inputs, parr_targets))
	tf_test = tf_test.shuffle(1000).batch(config.BATCH_SIZE)

	return tf_data, tf_test #(data_train, [labels_train, data_train]), (data_test, [labels_test, data_test])



def load_train_data(config):
	print("Loading numerai train data...")
	train_data = pd.read_parquet(config.DATA_PATH+config.NUMERAI_DATA_PATH+'numerai_training_data.parquet').fillna(0.5)

	feature_cols = [c for c in train_data if c.startswith("feature_")]
	target_cols = [c for c in train_data if c.startswith("target_")]

	era_vals = train_data["era"].unique()
	max_era = era_vals.max()

	word2id = {}
	for idx, name in enumerate(era_vals):
		word2id[name] = idx
	era_idx = train_data["era"].map(word2id)
	#era_idx = np.interp(era_idx, [0, max_era], [0, max_era//2]) #No era interpolation for now
	era_idx = np.rint(era_idx) / (len(word2id)-1) #Round down to nearest


	return train_data[feature_cols].to_numpy(), train_data[target_cols].to_numpy(), era_idx

def load_val_data(config):

	train_data = pd.read_parquet(config.DATA_PATH+config.NUMERAI_DATA_PATH+'numerai_validation_data.parquet').fillna(0.5)
	feature_cols = [c for c in train_data if c.startswith("feature_")]
	target_cols = [c for c in train_data if c.startswith("target_")]
	print(target_cols)
	print(train_data[target_cols])

	return train_data[feature_cols].numpy().astype(np.float32), train_data[target_cols].numpy().astype(np.float32), None #no era data for validation



def mse(y_true, y_pred):
	loss = tf.math.square(y_true - y_pred)
	loss = tf.reduce_mean(loss, axis=-1)
	return tf.reduce_mean(loss, axis=-1)

def train_model(config):
	keras.backend.clear_session()

	if os.path.exists(config.MODELS_PATH+config.MODEL_NAME):
		model = load_model(config)
	else:
		model = build_model(config)

	############################
	#We need to build a seperate model object for training due to a save/load bug in h5py
	input_1, input_2 = model.input
	model_targets = model.get_layer("targets").output 
	model_features = model.get_layer("features").output

	train_model = tf.keras.Model([input_1, input_2], [model_features, model_targets], name="model")

	train_model.compile(loss={"features":keras.losses.MeanSquaredError(), "targets":keras.losses.MeanSquaredError()}, loss_weights={"targets":10, "features":1}, optimizer=keras.optimizers.Adam(learning_rate=config.LR, beta_1=config.BETA_1, beta_2=config.BETA_2))

	tf_data, tf_test = build_tf_train(config, *load_train_data(config))

	from keras.callbacks import CSVLogger
	csv_logger = CSVLogger(config.MODELS_PATH+'log.csv', append=True, separator=',')

	#Load new data and train for config.EPOCHS epochs
	for i in range(0, config.EPOCHS):
		print(f"Train epoch {i}...")
		train_model.fit(tf_data, validation_data=tf_test, callbacks=[csv_logger])
		
		#Save the original graph
		model.save(config.MODELS_PATH+config.MODEL_NAME)
		if (i % 5) == 0:
			model.save(config.MODELS_PATH+f"epoch_{i}_"+config.MODEL_NAME)
		print("Model saved.")




def build_synthetic_feat_wrapper(config):
	build_synthetic_features(config, "training")
	build_synthetic_features(config, "validation")
	build_synthetic_features(config, "tournament")

def build_synthetic_features(config, numerai_data):
	#Saves new train and val data with synthetic features
	#numerai_data: "training", "validation", or "tournament"
	from data_manager import ERA_COL
	from data_manager import get_dictionaries

	print(f"reading {numerai_data} data from local file")

	train_data = pd.read_parquet(config.DATA_PATH+config.NUMERAI_DATA_PATH+f'numerai_{numerai_data}_data.parquet').fillna(0.5)
	feature_cols = [c for c in train_data if c.startswith("feature_")]
	target_cols = [c for c in train_data if c.startswith("target_")]

	new_stocks = [f"feature_synth_{i}" for i in range(config.SYNTH_STOCKS)]

	model = get_special_model(config)

	rand_eras = np.random.randint(0, high=575, size=config.SYNTH_STOCKS, dtype=int) / 575 #config.SYNTH_FEATURES-1
	rand_stocks = np.random.randint(0, high=5, size=(config.SYNTH_STOCKS, len(feature_cols)), dtype=int)

	print(rand_eras)
	print(rand_stocks)

	new_stocks, new_targets = model.predict([rand_stocks, rand_eras], batch_size=100)

	#Transformer new stocks into uniform 5-bin distribution
	if False:
		order = new_stocks.argsort()
		ranks = order.argsort().astype(np.int32)
		print(ranks)
		new_stocks = (np.digitize(ranks, [-0.1, 0.2*config.NUM_FEATURES, 0.4*config.NUM_FEATURES, 0.6*config.NUM_FEATURES, 0.8*config.NUM_FEATURES, config.NUM_FEATURES], right=True)-1) / 4
		print(print(np.histogram(new_stocks[0], bins=5)))
		print(new_stocks)
	else:
		new_stocks /= 4

	#new_stocks = new_stocks / 4
	new = np.concatenate([new_stocks, new_targets], axis=1)

	new_stocks_df= pd.DataFrame(new, columns=feature_cols+target_cols, dtype=np.float32)

	print("----------- NEW FEATURES:")
	print(new_stocks_df)


	new_stocks_df.to_pickle(config.SYNTH_FEAT_PATH+f"{numerai_data}.pkl")
	#new_train_feat.to_parquet(config.SYNTH_FEAT_PATH+"synth_feat_train.parquet") #Not working
	print(f"Completed stock generation on {numerai_data}.")


def get_special_model(config):
	#Builds a model with various output layer possibilities
	#Used for synthetic inference
	keras.backend.clear_session()

	model = load_model(config)
	input_1, input_2 = model.input

	if True: #config.SYNTH_METHOD == "quick":

		layer_output = model.get_layer("features").output
		targets_output = model.get_layer("targets").output

		special_model = tf.keras.Model([input_1, input_2], [layer_output, targets_output], name="special_model")
		special_model.summary()

	return special_model
