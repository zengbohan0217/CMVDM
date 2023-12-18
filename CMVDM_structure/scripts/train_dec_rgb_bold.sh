gpu=2  # GPU ID
tensorboard_log_dir=./log

echo src/train_decoder_bold.py --exp_prefix bold_rgb_only_0506_noDE_strd2_largek \
--enc_cpt_name bold_rgb_only_best --separable 1 --test_avg_qntl 1 --learning_rate 6e-3 --loss_weights 1,1,0 \
--fc_gl 1 --gl_l1 40 --gl_gl 400 --fc_mom2 40 --l1_convs 1e-4 --tv_reg 3e-1 --n_epochs 60 --batch_size_list 64,16,48,48 \
--scheduler 1 --percept_w 10,10,10,10,2 --rgb_mae 1 --is_rgbd 0 --norm_within_img 1 \
--depth_from_rgb 0 --tensorboard_log_dir $tensorboard_log_dir --gpu $gpu --may_save 1
