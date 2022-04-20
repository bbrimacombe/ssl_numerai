For the Numerai Self-Supervised competition, I tackled the problem of creating synthetic features using a deep autoencoder.

The autoencoder takes two inputs: the features for a single row, and the era number.
The autoencoder does two kinds of augmentation to the inputs:

1.) Maps them through a randomly initialized, frozen deep network. This is called extreme learning (used here for improved generalization).

2.) Concatenates the original features with 0.3 dropout to the “extreme” features.
The model encodes these inputs to a 12-dimensional latent space. Then it decodes the latent back to the full original feature space and is scored with mean squared error.

I train only on train-data eras.
I found it improved generalization to linearly interpolate eras from [0, ~550] down to [0, 12]. For example, era 200 will become era 5. During validation, era is set to the max value seen during training.

Once this model is trained, I create new features in two ways:

1.) Use the 12 dimensional latent space as new features.

2.) Use the argmax of the 12 dimensional latent space as a feature. As the data is not i.d.d., this is used to improve the generalization of our representation when out-of-sample. 

Combining the original feature-space with the argmax feature improves the baseline LightGBMR model sharpe by almost 3% on validation. Full results can be found here:
https://forum.numer.ai/t/numerai-self-supervised-learning-data-augmentation-projects/5003/15


##################################################

el_driver contains the main method to interface with el_features_model. Used to build the unsupervised model for synthetic features. Test them in the modified numerai example_model script. 

el_features_model_stocks is a fork of el_features_model used to generate synthetic stocks and targets from uniform noise.
