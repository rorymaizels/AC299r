
# use following command to access files in another directory
# sys.path.insert(0, "path")

import pyro_vae

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# set up the parameters for the data, model and training

data_params = {
    "dataset"           :   "BLAT_ECOLX"
    }

# commented out parameters are for unimplemented components
model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "learning_rate"     :   1.0e-3,
    # "logit_p"           :   0.001,
    # "sparsity"          :   "logit",
    # "final_decode_nonlin":  "sigmoid",
    # "final_pwm_scale"   :   True,
    # "n_pat"             :   4,
    "r_seed"            :   12345,
    # "conv_pat"          :   True,
    # "d_c_size"          :   40
    }

train_params = {
    "no_workers"        :   8,
    "num_updates"       :   300000,
    # "save_progress"     :   True,
    # "verbose"           :   True,
    # "save_parameters"   :   False,
    }

# speeds up training
cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

pyro.enable_validation(True)
pyro.set_rng_seed(model_params['r_seed'])
pyro.clear_param_store()

enc_arch = [model_params['encode_dim_zero'],model_params['encode_dim_one']]
dec_arch = [model_params['decode_dim_zero'],model_params['decode_dim_one']]

# creat data params
loader, alph, seq = pyro_vae.loader_function(data=data_params["dataset"],
                                            bs=model_params['bs'],
                                            nw=train_params['no_workers'],
                                             pm=cuda)

# construct model
vae = pyro_vae.VAE(alph_size=alph,seq_len=seq,z_dim=model_params['n_latent'],
                   encoder_architecture=enc_arch,decoder_architecture=dec_arch,use_cuda=cuda)

# define optimizer
optimizer = Adam({"lr": model_params['learning_rate']})

# construct stochastic variational inference module
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

NUM_EPOCHS = train_params['num_updates']
PRINT_FREQUENCY = 100

train_elbo = []
# training loop
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = pyro_vae.train(svi, loader, use_cuda=cuda)
    train_elbo.append(-total_epoch_loss_train)
    if epoch % PRINT_FREQUENCY == 0:
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

#torch.save(vae.state_dict(), './test_model.pt')


