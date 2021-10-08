CUDA_VISIBLE_DEVICES=0 \
python ../crossmpi_from_images.py \
--image1='/media/Data2/zhouyuemei/Giga_setting/20191228test/global_colored_002.png' \
--image2='/media/Data2/zhouyuemei/Giga_setting/20191228test/local_002.png' \
--output_dir='/media/NASzym/cvpr21/RealData/giga_setup' \
--image_width 1000 \
--image_height 750 \
--new_image_width 992 \
--new_image_height 736 \
--baseline 120 \
--intrinsic1 '21.677987421383648 29.26033519553073 1.518658280922432 0.6073184357541902' \
--intrinsic2 '21.504424999999998 28.901166666666665 0.44985 0.5209933333333333' \
--min_depth 30000 \
--max_depth 80000 \
--pose1 '1 0 0 0 0 1 0 0 0 0 1 0' \
--pose2 '0.9988190734077385 -0.012931066769675639 0.04683210553781299 117.595 0.012786934883065173 0.999912544906845 0.0033759197580984796 2.75014 -0.04687166407546157 -0.0027730939607339493 0.998897070301381 6.46072' \
--model_root '/media/NASzym/cvpr21/CrossMPI_models' \
--model_name 'AttentionRefSR_pretrain_hr_syn_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrained_plus_fuse_net' \
--which_model_predict 'guided_nnup_denseASPP_lowres_image_psv_lowresL1' \
--render_multiples 0 \
--render