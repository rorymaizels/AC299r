## Contents

- [Abstract](index.html)
- [Project Motivation](motivation.html)
- [Biological & Theoretical Background](background.html)
- [Model Structure](structure.html)
- [Usage](usage.html)
- [First Steps: Pyro](pyro.html)
- [Model Reconstruction](model.html)
- [Performance Comparison](performance.html)
- [Conclusions](conclusions.html)

# Model Structure

## Data Structure

Detailed descriptions of how data was collected and processed can be found in the Marks lab's papers describing their [EVmutation](https://www.nature.com/articles/nbt.3769) and [DeepSequence](https://www.nature.com/articles/s41592-018-0138-4) models. For ease of model testing and comparison, datasets prepared by the lab were used exclusively for this project. This preparation involved the following process: For a particular protein of interest (the focus sequence of the data set), a multiple-sequence alignment was performed of the protein family in five iterations of the HMM homology search tool jackhmmer against the UniRef
database of nonredundant protein sequences. The threshold for inclusion is 80% coverage, as this was found to be a good value for ensuring that sufficient sequence alignments were generated but the sequences aligned were still sufficiently similar.

The biomolecule sequences found in databases have biases due to human sampling (as some phylogeny have been sequenced more frequently than other) and due to evolutionary sampling (as some proteins are over-represented in some groups of species). As such, before applying data to the model, a sequence re-weighting schedule is performed to reduce these biases. Each sequence weight is calculated as the reciprocal of the number of sequences within a given Hamming distance cutoff. The sample size, N is thus changed to be the sum of of these sequence weights. The cutoff for this re-weighting schedule has been found to work well as 0.2 (80% sequence identity) for non-viral biomolecules.

As will be discussed below, loading of prepared data sets and performance of this re-weighting scheme is done by the data helper class.

* * * 

## Code Structure

### pt_helper.py

Dependencies: numpy, torch

This module contains important classes and functions for data loading and mutation effect prediction. This module is largely unchanged from the original version - only changes made are to remove Theano dependence and facilitate Torch/GPU compatibility.

#### DataHelper Class

The main component of this module, responsible for loading data from datasets, calculating sequence weight. Configures the data set, performs one-hot encoding of amino acids, generates alignment and contains the functions required to perform ELBO calculations for log-ratio calculations used in mutation effect prediction.

#### gen_simple_job_string

Generates a basic job string specifying the key parameters of the model - used during model performance evaluation. Job string used for output and model file names

#### gen_job_string

Generates a detailed job string specifying model parameters, used for output and model file names.

----

### pt_model.py

Dependencies: numpy, scipy, torch

The module for construction of the model. All model classes inherit torch's nn.Module class and call `super(cls_name, self).__init__()` to add itself to this class. In doing so, all model classes can access each other, creating an easy-to-use modular structure for building models. In doing this, the Encoder class, for example, can be shared easily by both the MLE and SVI versions of the VAE.

#### set_tensor_environment

Function used by all model classes to set correct torch tensor type with respect to float-size and cuda/GPU capabilities (both of which specified in training.)

#### Encoder Class

sub-class used to define the encoder model for moth SVI and MLE forms of the VAE model. Recieves arguments from full VAE class specifying parameters. 

In `__init__` the encoder architecture is specified, with option for convolving the input layer, non-linear functions are set up and the weights of the layers are initialised in accordance with the original model (weights are glorot-normal initialised, biases are set to 0.1 apart from the output log-sigma bias, which is set to -5). `forward()` defines the forward pass, accepting data batch x and passing the data through the architecture to return `z_mu` and `z_logsig`.

#### DecoderMLE Class

sub-class for decoder where parameters are not subject to variational approximation (parameters determined through maximum likelihood estimation rather than stochastic variational inference). 

`__init__` creates architecture, initialises weights as per original model, sets up non-linear functions and accepts arguments passed from the full VAE model. In setting up the architecture, if so specified, will create sparsity parameters, a final temperature parameter, a convolutional layer for the output and a dropout layer. `forward()` accepts two arguments; the original input `x` which is used in calculations for the ELBO, and `z`, which is passed through the decoder architecture. If sparsity is used, `forward` takes sparsity parameters and tiles them to correct dimension. If passed sparsity argument is `'logit'` then the sigmoid of these parameters is applied to the final weights, otherwise the exponential is applied. If temperature or convolution are used, they are also applied here. Once the final output of the decoder is created, log(P(x|z)) is calculated. The forward pass returns the reconstruction of `x`, this log-probability value and the values of the final output.

#### DecoderSVI Class

sub-class for decoder that performs variational approximation on parameters. 

`__init__` accepts arguments from full VAE class, sets up parameters and architecture, initialises variables and non-linear functions to be used in forward pass. For each weight, mu and sigma weights and biases are specified rather than just weights themselves, and these are used to sample the used weights. Since the weights and biases used in the forward pass of the decoder are actually sampled from these learnt weights, the layers specified here serve only to act as containers for the weights that are learnt, as opposed to being the actual objects applied in the forward pass. Another key difference here is that two containers are constructed; first a list called `variational_param_identifiers` which stores an identifier key for all decoder parameters subject to variational parameters other than the sparsity parameters, for which loss is calculated separately. Second, `variational_param_name_to_sigma`, a dictionary mapping each parameter to a prior sigma value for loss calculations - the default for all prior sigmas is 1, this dictionary is created to allow flexibility in changing this.

The `sampler` function inherits the GPU-enabled torch random number generator constructed in the VAE class, and applies this in for the decoder's own reparameterisation trick function for sampling from a gaussian distribution. 

`forward()` accepts `x` and `z` as arguments, passes `z` through the model, tiling and applying sparsity parameters if specified, as well as convolutions and a final temperature parameter. The key difference is that for each step, the `sampler` function is used to create weights and biases sampled from the model's parameters, and these sampled values are what is applied to the data. Once the reconstruction of x has been created, log(P(x|z)) is calculated and the forward pass returns the reconstruction of `x`, this log-probability value and the values of the final output.

#### VAE_MLE Class

Class that constructs the full bayesian VAE model.

`__init__` adds the class to nn.Module, accepts parameters passed by user in training, sets up the correct torch tensor environment, and creates the encoder and decoder as per these parameters.

`sampler` is a function that, through the reparameterization trick, applies a torch random number generator to perform gaussian sampling.

`_anneal` is a function used to perform annealing if specified. This is where the KLD loss is down-scaled in early updates to allow the model to 'learn' the data first, increasing stability. Default is to not use annealing, and annealing is not used in performance comparisons.

`update` takes in the log(P(x|z)) value from the decoder, the z-mu and z-logsigma values returned by the encoder, the update number (for annealing should it be activated) and the effective sequence number. Calculates the latent-space loss. If sparsity is used, will calculate the sparsity loss according to the specific sparsity prior selected. Calculates the l2-regularisation penalty term. Applies annealing to the KLD loss if used, applies scaling as well should this be used (should the user want the model to focus on reconstruction rather than KLD loss - by default this is not applied). Finally, calculates the log(P(x)) value approximation based on ELBO. returns this ELBO, the log(P(x|z)) value, the regularization loss and the latent space loss.

`likelihoods` takes in a particular `x` and returns the log(P(x)) estimation for that `x`, used in mutation effect prediction. This and subsequent functions are used in prediction and testing rather than training.

`all_likelihood_components` takes in a particular `x` and returns the log(P(x)) estimation, the latent space KLD loss and the log(P(x|z)) value.

`recognize` takes in a particular `x` and returns the mu and log-sigma of the latent variable outputted by the model's encoder.

`get_pattern_activations` takes in a particular `x` and returns the output of the decoder's final layer from that x.

#### VAE_SVI Class

Class that constructs the full bayesian VAE model.

`__init__` adds the class to nn.Module, accepts parameters passed by user in training, sets up the correct torch tensor environment, and creates the encoder and decoder as per these parameters.

`KLD_diag_gaussians` is a function to calculate the KL divergence between two diagonal gaussians, used throughout the variational approximation loss calculations.

`sampler` is a function that, through the reparameterization trick, applies a torch random number generator to perform gaussian sampling.

`_anneal` is a function used to perform annealing if specified. This is where the KLD loss is down-scaled in early updates to allow the model to 'learn' the data first, increasing stability. Default is to not use annealing, and annealing is not used in performance comparisons.

`gen_KLD_params` iterates through all parameters identified by the `variational_param_identifiers` and, by accessing the mu and logsigma values, calculates the KL divergence loss, summing across all the variational parameters and returning this sum.

`gen_KLD_sparsity` calculates the same loss for the sparsity parameters, which must be calculated differently due to the sparsity constraints. Here, loss can be calculated differently depending on which sparsity prior is chosen - the continuous relaxation of a spike and slab prior is the default that is used for performance comparisons.

`update` takes in the log(P(x|z)) value from the decoder, the z-mu and z-logsigma values returned by the encoder, the update number (for annealing should it be activated) and the effective sequence number. applies the KLD loss of variational decoder parameters, along with the KLD loss of the latent space and the log(P(x|z)) value to determine the full ELBO loss value, which is returned, along with values for log(P(x|z)), the variational parameter loss and the latent variable loss.

`likelihoods` takes in a particular `x` and returns the log(P(x)) estimation for that `x`, used in mutation effect prediction. This and subsequent functions are used in prediction and testing rather than training.

`all_likelihood_components` takes in a particular `x` and returns the log(P(x)) estimation, the latent space KLD loss and the log(P(x|z)) value.

`recognize` takes in a particular `x` and returns the mu and log-sigma of the latent variable outputted by the model's encoder.

`get_pattern_activations` takes in a particular `x` and returns the output of the decoder's final layer from that x.

----
 
### pt_train.py

Module containing the main training function, as well as loading and saving functions.

#### save

accepts a model and filepath amongst arguments, and saves the model.

#### load

Loads weights from a given path to a given model.

#### train

Takes in an instance of the DataHelper and VAE model, along with training parameters. Sets up saving file path, embeddings (if the use of embeddings is chosen in the DataHelper) and sets up the Adam solver with the model's parameters and a learning rate of 0.001). Generates training loop, whereby for each loop a mini-batch is selected (with calculated sequence weights used to select the weights to recalibrate against biases) and then, if required, the model and data are transferred to GPU with cuda. The Adam solver's gradients are reset (as is required with Torch, due to accumulating gradients), the forward pass is performed, loss is found, backpropagation is performed and the solver updates parameters. Updated values are saved, printed or stored as specified and this process repeats for each epoch.

### run_ptmle.py / run_ptsvi.py

These are the scripts that should be called to run the model. In each, the above modules are imported; data, model and training parameters are specified in dictionaries, the DataHelper and relevant model are constructed. Parameters are printed to output, and then the model is trained - at the end, the model is saved. Due to the different model architectures, different scripts are used for the SVI and MLE versions of the model.

### SVI_mutation_analysis.py / MLE_mutation_analysis

Example scripts for performing mutation effect prediction - largely based off code from the original project. Model is constructed and loaded with saved parameters, and a function is defined and called to calculate the Spearman R value for comparison of the model's predictions with experimental data. Greater agreement between the model's predictions and the experimental data will lead to larger values of Spearman R. Although the code for this analysis was largely written before this project, the code will be supplied for reference in [model performance](performance.html)

## Key Changes

Here, we briefly highlight some of the main differences and developments that are present in the new model.

- Increased Modularity. Through the use of torch's nn.Module class, the encoder, decoder and VAE elements of the models have been defined as separate classes. This makes the model cleaner, more readable and more adjustable - for example, both the SVI and MLE versions of the model inherit the Encoder class, meaning that adjustments are instantly shared between models and do not need to be repeated.
- Increased concision. Through use of torch's machine learning functions and modules, the code is more concise - the full model takes up about half the number of lines of code. Again, this improves readability and manageability of the model which makes usage easier for new users
- Less hard-coding. The original model was almost entirely defined in standard Python code, depending on Theano largely just for GPU compatibility. The new model takes advantage of PyTorch's many built in functionalities to define the model. Not only does this make the model more interpretable, but it makes hyper-parameter exploration and adjustment significantly easier. For example, to change the original model's optimizer from Adam to Adadelta, for example, would require adjustments to a minimum of 25 lines of code - now, this can be done by changing only one line. 
- Increased documentation. It is important for the development of this model that new members of the lab can easily learn how it is constructed and how it is used. To this end, the degree of commenting and documentation in the model has been increased so that throughout the model it is more clear what is being done.
- More Pythonic structure. Without needing to use compiled code or symbolic tensors, model functions can be defined by standard Python functions rather than with the use of the compiled function call `theano.function()`.
- A data-loading bug that was present in the original DataHelper script was fixed to ensure proper functionality. This bug prevented the loading of pre-prepared datasets.