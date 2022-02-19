python train.py  -nt -is_debug -no_comet


# os.chdir(os.path.join(WORKDIR, PROJECT))

# BATCHSIZE = 512
# KFID = 0
# KN = 10
# ENCDIM = 128
# PQBIT=4
# QBIT=4
# WS=10000
# LDS=100000
# LR=8e-5
# ELAYER=7
# DLAYER=7
# NHEAD=16
# DEMODEL=544
# DDMODEL=544
# EFFD=2104
# DFFD=2104




# MINLOSS=0.085
# MODELNAME="CSIPlus_p4geenc128ff2104q4e7d7h16d544_kf0b512abs1w1000lr1e4dp0ema999tqbest"


# !python train.py -m kf  -kfid {KFID} -kn {KN} -ms {MODELNAME} -num {NUM}  -save  -save_best $USE_TPU   -n_es_epoch 20 -n_lr_warmup_step {WS} -n_init_epoch 0 -bs {BATCHSIZE}  -abs 1 -ema 0.999   -xla_procs 1  -train_quantize  \
# -lr {LR}  -n_lr_decay_step {LDS} -lr_decay_rate 0.2  -dp 0 -enc_dim {ENCDIM} -n_q_bit {QBIT} -d_eff {EFFD} -d_dff {DFFD} -n_e_layer {ELAYER} -n_d_layer {DLAYER} -n_ehead {NHEAD}  -n_dhead {NHEAD} -d_emodel {DEMODEL}  -d_dmodel {DDMODEL} \
# -activation gelu  -save_half -use_fp16  -es 1e-5 -min_loss {MINLOSS}  -epochs 57 # -n_epoch_step 10 
# !ls -ltrh ../data/{MODELNAME}_KF{KFID}/*