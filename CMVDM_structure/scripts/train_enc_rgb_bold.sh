gpu=0  # GPU ID
tensorboard_log_dir=./log

echo src/train_encoder_bold.py \
--exp_prefix bold_rgb_only_0429 \
--separable 1 --n_epochs 150 --learning_rate 2e-3 --cos_loss 0.3 --random_crop_pad_percent 3 --scheduler 10 --gamma 0.5 --scheduler 1 \
--fc_gl 1 --fc_mom2 10 --l1_convs 1e-4 --is_rgbd 0 --allow_bbn_detach 1 --train_bbn 0 --norm_within_img 1 --may_save 1 \
--tensorboard_log_dir $tensorboard_log_dir --gpu $gpu