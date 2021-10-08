CUDA_VISIBLE_DEVICES=1 \
python ./train.py \
--pretrained_hr_syn_checkpoint_dir '/media/Data1/zhouyuemei/models/AttentionRefSR/AttentionRefSR_pretrain_hr_syn_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrained' \
--checkpoint_dir '/media/Data1/zhouyuemei/models/AttentionRefSR' \
--cameras_glob '/media/Data1/zhouyuemei/cameras_accurate_timestamp/train/????????????????.txt' \
--image_dir '/media/Data1/zhouyuemei/images_accurate_timestamp/train' \
--valid_cameras_glob '/media/Data1/zhouyuemei/cameras_accurate_timestamp/test/????????????????.txt' \
--valid_image_dir '/media/Data1/zhouyuemei/images_accurate_timestamp/test' \
--experiment_name 'AttentionRefSR_pretrain_hr_syn_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrained_plus_fuse_net' \
--which_model_predict 'guided_nnup_denseASPP_lowres_image_psv_lowresL1' \
--which_loss vgg_lowresL1 \
--vgg_model_file '/home/zhouyuemei/code/AttentionRefSR_without_validation/models/imagenet-vgg-verydeep-19.mat' \
--resize_factor 8 \
--continue_train \
--max_steps 302050
