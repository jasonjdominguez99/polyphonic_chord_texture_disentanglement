import warnings
warnings.simplefilter('ignore', UserWarning)

from model import DisentangleVAE
from ptvae import (
    RnnEncoder, TextureEncoder, PtvaeEncoder, 
    PtvaeDecoder, RnnDecoder
)
from dataset_loaders import MusicDataLoaders, TrainingVAE
from dataset import SEED
from amc_dl.torch_plus import (
    LogPathManager, SummaryWriters, ParameterScheduler,
    OptimizerScheduler, MinExponentialLR,
    TeacherForcingScheduler, ConstantScheduler
)
from amc_dl.torch_plus.train_utils import kl_anealing
import torch
from torch import optim
import json


def get_hyperparameters(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)
    
    return args


def define_poly_chord_model(args, device):
    chd_encoder = RnnEncoder(
        args["chd_encoder_input_dim"],
        args["chd_encoder_hidden_dim"],
        args["chd_encoder_z"]
    )
    rhy_encoder = PtvaeEncoder(
        device=device, z_size=args["rhy_encoder_z"],
        max_pitch=args["rhy_encoder_max_pitch"] - 8,
        min_pitch=args["rhy_encoder_min_pitch"]
    )

    # rhy_encoder = TextureEncoder(256, 1024, 256)
    # pt_encoder = PtvaeEncoder(device=device, z_size=152)

    chd_decoder = RnnDecoder(z_dim=args["chd_decoder_z"])
    pt_decoder = PtvaeDecoder(
        note_embedding=None,
        dec_dur_hid_size=args["pt_decoder_hid_size"],
        z_size=args["pt_decoder_z"]
    )

    model = DisentangleVAE(
        args["name"], device, 
        chd_encoder, rhy_encoder, 
        pt_decoder, chd_decoder
    )

    return model


def main():
    model_config_path = "poly_chord_model_config.json"
    args = get_hyperparameters(model_config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parallel = args["parallel"] == "True"       # converting from json string to bool
    parallel = (
        parallel
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else False
    )

    # define model
    model = define_poly_chord_model(args, device)

    # data loaders
    data_loaders = (
        MusicDataLoaders.get_loaders(
            SEED,
            bs_train=args["batch_size"],
            bs_val=args["batch_size"],
            portion=args["loader_portion"],
            shift_low=args["loader_shift_low"],
            shift_high=args["loader_shift_high"],
            num_bar=args["loader_num_bar"],
            contain_chord=True
        )
    )

    log_path_mng = LogPathManager(args["readme_fn"])

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = MinExponentialLR(
        optimizer, gamma=args["sched_gamma"], minimum=args["sched_min"]
    )
    optimizer_scheduler = OptimizerScheduler(
        optimizer, scheduler, args["clip"]
    )

    writer_names = [
        'loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
        'kl_rhy', 'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss'
    ]
    #, 'chord', 'root', 'chroma', 'bass']
    tags = {'loss': None}
    summary_writers = SummaryWriters(
        writer_names, tags, log_path_mng.writer_path
    )

    tf_rates = [
        (args["tf_rate1_1"], args["tf_rate1_2"]),
        (args["tf_rate2_1"], args["tf_rate2_2"]),
        (args["tf_rate3_1"], args["tf_rate3_2"])
    ]
    tfr1_scheduler = TeacherForcingScheduler(*tf_rates[0])
    tfr2_scheduler = TeacherForcingScheduler(*tf_rates[1])
    tfr3_scheduler = TeacherForcingScheduler(*tf_rates[2])

    weights_scheduler = ConstantScheduler(
        [args["weight1"], args["weight2"]]
    )
    beta_scheduler = TeacherForcingScheduler(
        args["beta"], 0., f=kl_anealing
    )

    params_dic = dict(
        tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
        tfr3=tfr3_scheduler, beta=beta_scheduler,
        weights=weights_scheduler
    )
    param_scheduler = ParameterScheduler(**params_dic)

    # train model
    training = TrainingVAE(
        device, model, parallel, log_path_mng,
        data_loaders, summary_writers, optimizer_scheduler,
        param_scheduler, args["n_epoch"]
    )
    training.run()

if __name__ == '__main__':
    main()
