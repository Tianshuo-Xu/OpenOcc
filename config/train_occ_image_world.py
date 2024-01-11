grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 50
return_len_ = 15
return_len_train = 15
# load_from = 'out/occworld_debug/epoch_200.pth'
load_from = '/hpc2hdd/home/txu647/code/OccWorld/out/vqvae/epoch_103.pth'
port = 25097
revise_ckpt = 3
mini_batch = 2
eval_every_epochs = 5
save_every_epochs = 5
multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False)

freeze_dict = dict(
    vae = True,
    img_vae = True,
    transformer = False,
    pose_encoder = False,
    pose_decoder = False,
)
optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        weight_decay=0.01,
    ),
)

img_exist = True

data_path = '/hpc2hdd/home/txu647/data/nuscenes/'

train_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_train+1, 
    offset = 0,
    imageset = '/hpc2hdd/home/txu647/data/nuscenes_infos_train_temporal_v3_scene.pkl', 
)

val_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_+1, 
    offset = 0,
    imageset = '/hpc2hdd/home/txu647/data/nuscenes_infos_val_temporal_v3_scene.pkl', 
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='train', 
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='val', 
)

train_loader = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 8,
)
    
val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 4,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='CeLoss',
            weight=1.0,
            input_dict={
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'}),
        dict(
            type='MSELoss',
            weight=6.0,
            input_dict={
                'mse_img_inputs': 'mse_img_inputs',
                'mse_img_labels': 'mse_img_labels'}),
        dict(
            type='PlanRegLossLidar',
            weight=0.1,
            loss_type='l2',
            num_modes=3,
            input_dict={
                'rel_pose': 'rel_pose',
                'metas': 'metas'})
    ]
)

loss_input_convertion = dict(
    ce_inputs = 'ce_inputs',
    ce_labels = 'ce_labels',
    mse_img_inputs = 'mse_img_inputs',
    mse_img_labels = 'mse_img_labels',
    rel_pose='pose_decoded',
    metas ='output_metas',
)

base_channel = 64
image_channel = 24
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type = 'TransVQVAE',
    num_frames=return_len_,
    delta_input=False,
    offset=1,
    vae = dict(
        type = 'VAERes2D',
        encoder_cfg=dict(
            type='Encoder2D',
            ch = base_channel, 
            out_ch = base_channel, 
            ch_mult = (1,2,4), 
            num_res_blocks = 2,
            attn_resolutions = (50,), 
            dropout = 0.0, 
            resamp_with_conv = True, 
            in_channels = _dim_ * expansion,
            resolution = 200, 
            z_channels = base_channel * 2, 
            double_z = False,
        ), 
        decoder_cfg=dict(
            type='Decoder2D',
            ch = base_channel, 
            out_ch = _dim_ * expansion, 
            ch_mult = (1,2,4), 
            num_res_blocks = 2,
            attn_resolutions = (50,), 
            dropout = 0.0, 
            resamp_with_conv = True, 
            in_channels = _dim_ * expansion,
            resolution = 200, 
            z_channels = base_channel * 2, 
            give_pre_end = False
        ),
        num_classes=18,
        expansion=expansion, 
        vqvae_cfg=dict(
            type='VectorQuantizer',
            sane_index_shape=True,
            n_e = n_e_, 
            e_dim = base_channel * 2, 
            beta = 1., 
            z_channels = base_channel * 2, 
            use_voxel=False)),
    
    transformer=dict(
        type = 'PlanUAutoRegTransformer',
        num_tokens=1,
        num_frames=return_len_,
        num_layers=2,
        occ_shape=(base_channel * 2 + image_channel, 50, 50),
        pose_shape=(1, base_channel * 2 + image_channel),
        pose_attn_layers=2,
        pose_output_channel=base_channel * 2 + image_channel,
        tpe_dim=base_channel * 2 + image_channel,
        channels=(
            base_channel * 2 + image_channel, 
            base_channel * 4 + image_channel * 2, 
            base_channel * 8 + image_channel * 4
        ),
        temporal_attn_layers=6,
        output_channel=n_e_ + image_channel,
        learnable_queries=False,
        img_vae_exist=img_exist,
    ),
    pose_encoder=dict(
        type = 'PoseEncoder',
        in_channels=5,
        out_channels=base_channel * 2 + image_channel,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    pose_decoder=dict(
        type = 'PoseDecoder',
        in_channels=base_channel * 2 + image_channel,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    img_vae_exist=img_exist,
)


shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
