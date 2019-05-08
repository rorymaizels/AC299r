# Project Motivation

[DeepSequence](https://github.com/debbiemarkslab/DeepSequence) is a deep, generative model designed to capture the latent structure of biomolecule sequence families and exploit this structure for mutation effect prediction (for more details on the theoretical background of this idea see the [background page](background.md)). It is a very large model with multiple different regularization and architectural features for improved performance - and has biologically motivated priors built into the model (for a full explanation of the model and code structure, see the [model structure page](structure.md)) - in their entirety, the Python scripts required to build, train and test the DeepSequence model consist of over two thousand lines of code. Many aspects of the model, such as the architecture, optimisation and loss functions, are hard coded in standard Python code, using the machine learning library *Theano* to provide GPU capabilities.

Theano is a Python library used for evaluating mathematical expression that compiles to efficiently run on either CPU or GPU architectures. Through Theano's symbolic tensor capabilities, one can construct computational graphs for machine learning models that are efficiently compiled and run. As such, Theano has been a popular library for machine learning since its initial release in 2007 - however, [in September 2017](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ), it was announced that Theano would no longer be actively developed due to the strength of competing products such as Pytorch and Tensorflow, that were being developed by major industrial players (Facebook and Google respectively). 

As such, if the library is not already effectively dead, it is dying. The most [up-to-date](http://deeplearning.net/software/theano/requirements.html) version of theano does not support Python 3.7, nor the most recent versions of NumPy, SciPy, GCC or CUDA (the former two being integral libraries for mathematical and statistical programming, the latter two being integral to GPU-based computing). Currently, this may only be an inconvenience that can be resolved with careful environment construction, however eventually models built in Theano will be obsolete. Moreover, Theano implementations cannot improve and will not gain access to the addition of better functionalities and supporting packages, as would be the case with models constructed with more modern libraries. As such, it is important for the longevity of the DeepSequence model that it be rebuilt with a more modern library.

Reasons for reimplementing the model extent beyond Theano's expiration. Pytorch is a popular machine learning library that is fast, intuitive to use and rapidly growing in functionality. Pytorch's 'U.S.P.' is that it defines computational graphs dynamically. Theano and Tensorflow both implement static computational graphs - you construct your model's computational graph first, using symbolic tensors as placeholders for the actual values, and once everything is constructed and compiled you run your model. PyTorch, however, constructs the computational graph 'on the fly' over each forward pass of the model. This means that PyTorch models are afforded  far greater flexibility - model structure can be changed for each forward run, should the user so wish. Additionally, the design philosophy of PyTorch is often thought to be more intuitive, pythonic and simple to learn. Moreover, by removing the need for symbolic tensors and graph compilation, debugging code issues and visualising intermediate values is considerably more straightforward in PyTorch - this is especially apparent with custom implementations. Finally, PyTorch offers considerable and ever-increasing support from add-on libraries for specific needs and requirements. For probabilistic programming, there is Uber's [pyro](https://pyro.ai/) package. For Gaussian Processes, there is [GPyTorch](https://gpytorch.ai/). Just this month, PyTorch released two new supporting libraries, [BoTorch](https://botorch.org/) and [Ax](https://ax.dev/), for Bayesian Optimisation and adaptive experimentation respectively. 

Thus, reconstructing this model in PyTorch provides the model with greater flexibility for future improvements, opens access to the use of new machine learning libraries, and makes usage and development quicker and easier for members of the lab. 

Reimplementing this model also presented considerable personal value. It offered me an intensive and research driven education in a number of areas of which I had no prior experience:
- Programming with Theano, PyTorch and Pyro.
- Bayesian variational autoencoders and stochastic variational inference.
- Building large-scale, custom implemented machine learning models with biologically motivated priors.

## Contents

- [Abstract](index.md)
- [Project Motivation](motivation.md)
- [Biological & Theoretical Background](background.md)
- [Model Structure](structure.md)
- [Usage](usage.md)
- First Steps: Pyro
- Model Reconstruction
- Performance Comparison
- [Conclusions](conclusions.md)





