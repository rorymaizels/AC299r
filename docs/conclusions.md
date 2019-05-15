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

# Conclusion

DeepSequence is a deep, generative latent-variable model that can be used to predict the effect of mutations on a protein's function by leveraging evolutionary sequence variation information. The original model was constructed in Theano, a mathematical and machine learning library that is no longer being actively developed. The aim of this project was to reconstruct this complex model in its entirety with a more modern library in the hope of facilitating continued use and development of the model. To do this, after exploring options such as PyTorch's Pyro library, I implemented this model using a class-based PyTorch structure.

The original plan for this project was to develop the model further beyond reimplementation. Ultimately, the project presented numerous challenges throughout the implementation process - in particular, developing a working understanding of Theano, Pytorch, variational autoencoders and stochastic variational inference largely from scratch presented a particular challenge that slowed the progress of the project. Upon construction of the new model, considerable time was dedicated to debugging and inspecting to probe how, exactly, the original and new models behaved.

The model constructed through this project has been found to learn properly and successfully perform predictions that can be compared to experimentally determined values - however, it is clear that the model's performance is not exactly the same as the original implementation. The first authors of the original model have since left the Marks lab, meaning that opportunities to discuss the exact details of the original implementation have been limited.

There are various possibilities for the cause of this difference in performance. Inspection of the model's performance shows that model features such as convolution, sparsity, temperature are being implemented, but it is possible that there are subtle differences in implementation. For example, the model contains multiple non-standard weight manipulation steps, such as the tiling and subsequent application of sparsity weights. The different framework for forward passes in the two libraries (Theano's static, symbolic graph compared to Torch's dynamic, on-the-fly graph) means that implementation of these manipulations may differ slightly in non-obvious ways. It is also possible that the torch implementation of Adam differs subtly from the custom defined Adam function in the original model. With the size, complexity and stochastic nature of the model, determining the exact behaviour of model features requires careful analysis that, unfortunately, could not be done in time for the project deadline. 

Future work will initially focus on improving performance and synchronizing behaviour as much as possible with the original model. There are numerous steps to achieving this: the models can be examined in a deterministic fashion, removing randomness and stochasticity (removing sampling, setting random seeds carefully) to check if mathematic operations match up as they appear to - additionally, loading weights from one model into the other will help determine whether the issue is just of the weights that are learnt or whether there is a deeper mathematical issue. A further comprehensive inspection of the performance of all Theano functions and the PyTorch functions with which they are replaced will be required to ensure that all behaviours here are compatible and alike. A particularly close inspection of the weights and intermediate values produced in tiling, sparsity and convolution operations would likely surface underlying differences in how the models' graphs might be structured. Alongside this, running models multiple times would prove valuable for determining to what extent the model's outputs and performances differ.

Some aspects of the original model were chosen emperically, as is usually the case with deep learning models. Thus, in the case that this model's behaviour is subtly different but not flawed, empirical exploration of model parameters could be sufficient to bring about sufficient improvements in performance.

Aside from this behavioural aspect of model tweaking, one additional implementation to do in the future is to generalise model architecture: currently, for ease of construction, the model is only compatible with two hidden layers in each network. Generalising the models to accept variable hidden layer dimensions would be important for allowing hyperparameter tuning.

Once the model's behaviour has been adjusted, the new model is in an excellent position for substantial improvements. Focussing on speed-up, implementation of torch modules such as `torch.nn.DataParallel` for data parallelism and implementation of a custom-defined, GPU enabled data loader should see performance speed increase substantially. 

The modular nature of the new model facilitates easy hyper-parameter tuning and exploration. Moreover, during construction of this model, PyTorch released 'Ax', a library for adaptive exploration of model and experiment hyper-parameters. This library can implement bayesian optimization and bandit optimization to find the best set of parameters for model performance. In the context of DeepSequence, which has a huge hyper-parameter space, this could be valuable in optimizing performance as much as possible.
 
Whilst the work on this PyTorch implementation of DeepSequence is not finished, I have already gained a huge amount from this experience. Apart from the obvious initial exposures to stochastic variational inference, latent-variable models, Theano and PyTorch programming, the process of constructing, testing, and debugging a large and sophisticated computational model has provided a challenging and rewarding experience.

```
        ____                 _____
       / __ \___  ___  ____ / ___/___  ____ ___  _____  ____  ________
      / / / / _ \/ _ \/ __ \\__ \/ _ \/ __ `/ / / / _ \/ __ \/ ___/ _ \
     / /_/ /  __/  __/ /_/ /__/ /  __/ /_/ / /_/ /  __/ / / / /__/  __/
    /_____/\___/\___/ .___/____/\___/\__, /\__,_/\___/_/ /_/\___/\___/
                   /_/                 /_/
```


