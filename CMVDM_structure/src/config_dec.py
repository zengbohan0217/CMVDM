from src.config import *
Train_Decoder = True

if Train_Decoder:
    flags.DEFINE_string("tensorboard_log_dir", '', "Log dir.")
    flags.DEFINE_string("exp_prefix", 'dec_tmp', "Experiment prefix.")
    flags.DEFINE_integer("num_workers_gpu", 2, "Number of workers per GPU.")
    flags.DEFINE_enum("roi", 'V2', ['V1', 'V2', 'V3', 'V4', 'FFA', 'PPA', 'LOC', 'LVC', 'HVC', 'VC', 'V1V2V3'], '')
    flags.DEFINE_integer("n_epochs", 150, "Number of epochs.")
    flags.DEFINE_float("learning_rate", 5e-4, "The initial value for the learning rate.")
    flags.DEFINE_integer("scheduler", 0, "Reduce learning rate by scheduler.")
    flags.DEFINE_float("decay", 0., "Weight decay.")
    flags.DEFINE_string('enc_cpt_name', '', 'Encoder checkpoint name to load')
    flags.DEFINE_string('enc_bbn_arch_name', 'alexnet', '')
    flags.DEFINE_list("batch_size_list", [24, 16, 16, 64], "Supervised training | unlabeled fMRI | unlabeled images | test")
    batch_size_list = lambda : [int(x) for x in FLAGS.batch_size_list]
    flags.DEFINE_list("loss_weights", [1, 1, 1], "Loss weights [loss_D, loss_ED, loss_DE]")
    flags.DEFINE_list("percept_w", [1, 1, 1, 10, 1], "Perceptual loss weights along blocks")
    flags.DEFINE_float("test_avg_qntl", 1., "Quantile of test repeats to be averaged at random")
    flags.DEFINE_float("mix_q", 0, "Quotient of mixed samples")
    flags.DEFINE_list("mslr", [20, 35, 45, 50], "Scheduler milestones.")
    flags.DEFINE_float("sched_gamma", .2, "Scheduler lr reduction gamma")
    flags.DEFINE_integer("n_conv_layers", 3, "Number of convolutional layers in Decoder model.")
    flags.DEFINE_integer("sum_level", 6, 'Summary level. 4: -BatchOutDist | 5: Report all. | 6: +ImageLoss components')
    flags.DEFINE_list("train_montage_indices", list(np.linspace(0, 1199, 50, dtype='int')), '')
    flags.DEFINE_enum('interp_mode', 'bicubic', ['nearest', 'bicubic', 'bilinear'], '')
    flags.DEFINE_integer("pred_interp", 0, "Size to interpolate the reconstructed image before applying loss. '0' means no interp")
    flags.DEFINE_float("tv_reg", 0.5, "Total variation regularization coefficient.")
    flags.DEFINE_float("ml_percep", 1., "Multi-layer perceptual loss.")
    flags.DEFINE_integer("random_crop_pad_percent", 3, "")
    flags.DEFINE_integer("config_train", 10, "Training configuration. 1: Decoder supervised training only | 10: Full method")
    flags.DEFINE_float("l1_fcreg", 0., "")
    flags.DEFINE_float("l1_convs", 0., "")
    flags.DEFINE_float("l2_convs", 0., "")
    flags.DEFINE_float("l2_fcreg", 0., "")
    flags.DEFINE_float("fc_mom2", 0., "")
    flags.DEFINE_float("fc_gl", 0, "")
    flags.DEFINE_float("gl_l1", 20, "L1 reg component of GL")
    flags.DEFINE_float("gl_gl", 400, "GL component of GL")
    flags.DEFINE_float("vox_nc_reg", 0., "")
    flags.DEFINE_float("rgb_mae", 0.2, "")
    flags.DEFINE_integer('verbose', 1, '0: -within-epoch log | 1: Report all.')


exp_folder = None
exp_folder = lambda : os.path.join('results/', FLAGS.exp_prefix)
enc_cpt_path = lambda : f'{PROJECT_ROOT}/checkpoints/{FLAGS.enc_cpt_name}.pth.tar'

if __name__ == '__main__':
    pass
