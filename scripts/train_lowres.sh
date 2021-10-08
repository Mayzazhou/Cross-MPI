CUDA_VISIBLE_DEVICES=1 \
python ./train.py \
--checkpoint_dir '/media/Data2/zhouyuemei/models/AttentionRefSR' \
--cameras_glob '/media/Data2/zhouyuemei/cameras_accurate_timestamp/train/????????????????.txt' \
--image_dir '/media/Data2/zhouyuemei/images_accurate_timestamp/train' \
--valid_cameras_glob '/media/Data2/zhouyuemei/cameras_accurate_timestamp/test/????????????????.txt' \
--valid_image_dir '/media/Data2/zhouyuemei/images_accurate_timestamp/test' \
--experiment_name 'AttentionRefSR_pretrain_hr_syn_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrain' \
--which_model_predict 'guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrain' \
--which_loss pixel \
--vgg_model_file '/home/zhouyuemei/code/AttentionRefSR_without_validation/models/imagenet-vgg-verydeep-19.mat' \
--resize_factor 8 \
--pretrain_hr_syn \
--pretrain_lowres \
--max_steps 817000
