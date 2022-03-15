import os 
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

#optional
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

def build_model(config):
	#construct keras model graph 

	input_1 = layers.Input((config.NUM_FEATURES), dtype=tf.int8) 
	input_1_cast = layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32))(input_1)
	input_2 = layers.Input((1,), dtype=tf.float32)
	
	kernel_initializer = keras.initializers.RandomNormal(stddev=0.05)
	bias_initializer = keras.initializers.RandomNormal(stddev=0.05)
	model_1 = layers.Dense(100, activation=keras.activations.relu, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(input_1_cast) #1000
	model_1 = layers.Dense(1000, activation=keras.activations.sigmoid, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(model_1)
	model_1 = layers.Lambda(lambda x: K.stop_gradient(x))(model_1)

	input_1_cast = layers.Dropout(0.3)(input_1_cast, training=True) #training=True bc we want test set information to be same

	input_1_cast = layers.Concatenate(axis=-1)([input_1_cast, input_2, model_1])

	#Training = true for less functional drift during inference
	itermediate1 = layers.Dense(1000, activation="sigmoid", name="dense0")(input_1_cast)
	itermediate1 = layers.Dropout(0.2)(itermediate1, training=True)
	itermediate1 = layers.Dense(1000, activation="sigmoid", name="dense1")(itermediate1)
	itermediate1 = layers.Dropout(0.2)(itermediate1, training=True)

	#typo :p "itermediate"
	itermediate = layers.Dense(12, activation="sigmoid", name="itermediate")(itermediate1) #, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)

	itermediate2 = layers.Dense(1000, activation="sigmoid", name="dense3")(itermediate)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)
	itermediate2 = layers.Dense(1000, activation="sigmoid", name="dense4")(itermediate2)
	itermediate2 = layers.Dropout(0.2)(itermediate2, training=True)


	features = layers.Dense(config.NUM_FEATURES, activation="sigmoid")(itermediate2)
	features = layers.Lambda(lambda x: 4*x, name="features")(features) #To span range (integer targets)

	model = tf.keras.Model([input_1, input_2], [features], name="model")

	print("Built model:")
	model.summary()

	return model

def load_model(config):
	#tf.keras.mixed_precision.set_global_policy('mixed_float16')
	model = tf.keras.models.load_model(config.MODELS_PATH+config.MODEL_NAME) #+".h5"
	model.summary()

	return model


def build_deterministic_partitions(config, data):

	return (4*data).astype(np.int16)

	"""
	#For now just some quick heuristics
	#data must be a numpy array of shape dimension 2
	first_third = data[:, :config.NUM_FEATURES//3]
	second_third = data[:, config.NUM_FEATURES//3:2*config.NUM_FEATURES//3]
	third_third = data[:, 2*config.NUM_FEATURES//3:]
	stride_four = data[:, ::4]
	stride_seven = data[:, ::7]

	print("---------------------------------")
	print(first_third)
	print(second_third)
	print("---------------------------------")
	return first_third, second_third, third_third, stride_four, stride_seven #data[:, :100] # data #
	"""


def build_tf_train(config, x, y, eras):

	#Map data to integers
	feat = (4*x).astype(np.int16)

	print("Spliting train and test (skleran)...")
	from sklearn.model_selection import train_test_split
	data_train, data_test, era_train, era_test = train_test_split(feat, eras, test_size=0.1, random_state=0) #0.05

	print("feat------------------------")
	print(data_train)
	#parr_inputs = tf.data.Dataset.from_tensor_slices((build_deterministic_partitions(config, x)))
	parr_inputs = tf.data.Dataset.from_tensor_slices((data_train, era_train))
	parr_targets = tf.data.Dataset.from_tensor_slices((data_train)) 
	tf_data = tf.data.Dataset.zip((parr_inputs, parr_targets))
	tf_data = tf_data.shuffle(10000000).batch(config.BATCH_SIZE)


	parr_inputs = tf.data.Dataset.from_tensor_slices((data_test, era_test))
	parr_targets = tf.data.Dataset.from_tensor_slices((data_test))  
	tf_test = tf.data.Dataset.zip((parr_inputs, parr_targets))
	tf_test = tf_test.shuffle(1000).batch(config.BATCH_SIZE)


	return tf_data, tf_test



def load_train_data(config):
	print("Loading numerai train data...")
	train_data = pd.read_parquet(config.DATA_PATH+config.NUMERAI_DATA_PATH+'numerai_training_data.parquet').fillna(0.5)

	#train_data = train_data[:10000]

	feature_cols = [c for c in train_data if c.startswith("feature_")]
	target_cols = [c for c in train_data if c.startswith("target_")]

	#Get era index and interpolate
	era_vals = train_data["era"].unique()
	max_era = era_vals.max()
	word2id = {}
	for idx, name in enumerate(era_vals):
		word2id[name] = idx
	era_idx = train_data["era"].map(word2id)
	era_idx = np.interp(era_idx, [0, max_era], [0, config.SYNTH_FEATURES-1]) #Interpolate to smaller scale
	era_idx = np.rint(era_idx).astype(np.int16) #Round down to nearest

	print(era_idx)
	print("-----")

	print(target_cols)
	print(train_data[target_cols])
	#train_data[feature_cols+target_cols] = train_data[feature_cols+target_cols].astype(np.int16)

	print(train_data[feature_cols].to_numpy().astype(np.float32))
	print("----------------------")
	print(train_data[target_cols].to_numpy().astype(np.float32))

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

	#Build model and compile
	input_1, input_2 = model.input
	model_features = model.get_layer("features").output
	train_model = tf.keras.Model([input_1, input_2], [model_features], name="model")
	train_model.compile(loss={"features":keras.losses.MeanSquaredError()}, optimizer=keras.optimizers.Adam(learning_rate=config.LR, beta_1=config.BETA_1, beta_2=config.BETA_2)) 

	tf_data, tf_test = build_tf_train(config, *load_train_data(config))

	from keras.callbacks import CSVLogger
	csv_logger = CSVLogger(config.MODELS_PATH+'log.csv', append=True, separator=',')

	#Load new data and train for config.EPOCHS epochs
	for i in range(0, config.EPOCHS):
		print(f"Train epoch {i}...")
		train_model.fit(tf_data, validation_data=tf_test, callbacks=[csv_logger])

		#(Save the original graph model)
		model.save(config.MODELS_PATH+config.MODEL_NAME)
		if (i % 5) == 0:
			model.save(config.MODELS_PATH+f"epoch_{i}_"+config.MODEL_NAME)
		print("Model saved.")




def build_synthetic_feat_wrapper(config):
	build_synthetic_features(config, "training")
	build_synthetic_features(config, "validation")
	build_synthetic_features(config, "tournament")

def build_synthetic_features(config, numerai_data):
	#Saves new train and val data with synthetic features concatenated along 1st axis
	#numerai_data: "training", "validation", or "tournament"
	from data_manager import ERA_COL
	from data_manager import get_dictionaries

	print(f"reading {numerai_data} data from local file")

	train_data = pd.read_parquet(config.DATA_PATH+config.NUMERAI_DATA_PATH+f'numerai_{numerai_data}_data.parquet').fillna(0.5)
	feature_cols = [c for c in train_data if c.startswith("feature_")]
	new_features = [f"feature_synth_{i}" for i in range(config.SYNTH_FEATURES)]

	model = get_special_model(config)

	train_data["era_feature"] = config.SYNTH_FEATURES-1

	new = model.predict([build_deterministic_partitions(config, train_data[feature_cols].to_numpy()), train_data["era_feature"].to_numpy()], batch_size=100)

	new_feat_df = pd.DataFrame(new, index=train_data.index, columns=new_features, dtype=np.float32)

	print("----------- NEW FEATURES:")
	for i in range(8):
		print(new_feat_df[new_features[i]])

	new_feat_df.to_pickle(config.SYNTH_FEAT_PATH+f"{numerai_data}.pkl")
	#new_train_feat.to_parquet(config.SYNTH_FEAT_PATH+"synth_feat_train.parquet") #Not working
	print(f"Completed feature generation on {numerai_data}.")


def get_special_model(config):
	#Builds a model with various output layer possibilities
	#Used for synthetic feature inference
	keras.backend.clear_session()

	model = load_model(config)
	#input_1, input_2, input_3, input_4, input_5 = model.input 
	input_1, input_2 = model.input

	if True: 

		layer_output = model.get_layer("itermediate").output
		#layer_output = model.get_layer("model_out").output

		#special_model = tf.keras.Model([input_1, input_2, input_3, input_4, input_5], [layer_output], name="special_model")
		special_model = tf.keras.Model([input_1, input_2], [layer_output], name="special_model")
		special_model.summary()


	return special_model
