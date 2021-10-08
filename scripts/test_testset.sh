CUDA_VISIBLE_DEVICES=0 \
python test.py \
--model_root '/media/NASzym/cvpr21/CrossMPI_models' \
--model_name 'AttentionRefSR_pretrain_hr_syn_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrained_plus_fuse_net' \
--vgg_model_file './models/imagenet-vgg-verydeep-19.mat' \
--data_split test_shuffle2_stride23 \
--num_runs 6546 \
--shuffle_seq_length 2 \
--cameras_glob '/media/Data1/zhouyuemei/cameras_accurate_timestamp/test/????????????????.txt' \
--image_dir '/media/Data1/zhouyuemei/images_accurate_timestamp/test' \
--output_root '/media/NASzym/cvpr21/CrossMPI_results' \
--which_model_predict guided_nnup_denseASPP_lowres_image_psv_lowresL1 \
--test_outputs ref_image_tgt_image_src_images \
--resize_factor 8