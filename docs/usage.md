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

# Usage

## Dependencies

- PyTorch ([installation guide here](https://pytorch.org/get-started/locally/?source=Google&medium=PaidSearch&utm_campaign=1712416206&utm_adgroup=67591282235&utm_keyword=%2Bpytorch%20%2Binstallation&utm_offering=AI&utm_Product=PYTorch&gclid=CjwKCAjwq-TmBRBdEiwAaO1en4MRL3TmS6kykIEfl0hsaWzdN_NDMkr4CGOTG8DKP99RPanh3hzRCxoCgbQQAvD_BwE))
- NumPy (through Anaconda or `pip install numpy`)
- SciPy (through Anaconda or `pip install scipy`)

## Running

The code for this model is included in this website's github, but a fully functional version will soon be deployed on the [Marks lab's github](https://github.com/debbiemarkslab). It is recommended that once the lab's version is available, that version should be downloaded.

1. Create Environment
	- using `conda create -n env_name python=x.x` create your environment for running the model. The model was tested using Python 2.7 but is compatible with Python 3. Either through a `.yaml` file or through terminal, ensure that the model's dependencies are installed.
3. Clone Repo
	- Acquire the necessary code by cloning the github repository.
2. Prepare Sequence Alignment Dataset
	- To use a sample data set from the Marks lab, run the download_alignments.sh file (`./download_alignments.sh` in terminal)
	- To construct a new sequence alignment data set, it is advised that one uses the [EVcouplings software package](https://github.com/debbiemarkslab/EVcouplings). This package was constructed by the Marks lab for a previous statistical model, and generates sequence alignments in the correct format for both EVmutation and DeepSequence packages.
3. Set model parameters
	- In `run_ptmle.py` or `run_ptsvi.py` for MLE- or SVI-VAE models respectively, set the data, model and training parameters in the respective dictionaries
4. Set up directory
	- Ensure all modules can access one another and that a 'model_params' folder has been created to store saved models.
5. Run!
	- For CPU running (which is strongly advised against), run `python run_ptxxx.py` where `xxx` is `mle` or `svi` depending on the model.
	- For GPU running with cuda, construct a script to run the model using `sbatch script_name.sh`. An example of a script to run this model is:

```
#!/bin/bash
#SBATCH -c 8
#SBATCH -t 18:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:4

module load gcc/6.2.0
module load cuda/9.0

python run_ptsvi.py
```


