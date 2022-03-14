#
###Main method for "exreme learning features model" or el_features_model.py
#

class config:

	###All file paths
	DATA_PATH = "data\\"
	NUMERAI_DATA_PATH = "numerai_datasets\\"
	MODELS_PATH = "models\\"
	MODEL_NAME = "el_feature_model_04.h5" #el_feature_model_00 has latent dim == 100
	SYNTH_FEAT_PATH = DATA_PATH+NUMERAI_DATA_PATH+"synth_feat6_"

	#### Dataset config
	NUM_FEATURES = 1050 
	NUM_TARGETS = 20

	#### Train config
	EPOCHS = 500
	BATCH_SIZE = 32 #32 #32
	LR = 0.00005 #0.0001
	BETA_1 = 0.95 #By default using adam 
	BETA_2 = 0.98

	### Synthetic features parameters
	SYNTH_FEATURES = 12



from el_features_model import train_model
from el_features_model import build_synthetic_feat_wrapper
def main():
	#train_model(config)
	build_synthetic_feat_wrapper(config)


if __name__ == '__main__':
	main()
