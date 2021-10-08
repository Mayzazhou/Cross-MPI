CUDA_VISIBLE_DEVICES=0 \
python ../test_giga.py \
--image_dir='/media/Data2/zhouyuemei/xiaoyun_giga/cam_05_66_cam_00_86_all_set_colour_transfered_quater' \
--output_dir='/media/NASzym/cvpr21/RealData/RealStereo' \
--frame_start=300 \
--frame_len=2 \
--fx=1.43577 \
--fy=1.91436 \
--min_depth 20 \
--max_depth 100 \
--pose1 1,0,0,0,0,-1,0,0,0,0,-1,0 \
--pose2 0.99975,-0.018756,0.01222,0.74008,-0.018748,-0.99982,-0.00075602,-0.026004,0.012232,0.00052673,-0.99993,-0.20932 \
--model_root '/media/NASzym/cvpr21/CrossMPI_models' \
--model_name 'AttentionRefSR_pretrain_hr_syn_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrained_plus_fuse_net' \
--which_model_predict 'guided_nnup_denseASPP_lowres_image_psv_lowresL1'