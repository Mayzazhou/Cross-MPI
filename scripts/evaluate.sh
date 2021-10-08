CUDA_VISIBLE_DEVICES=4 \
python evaluate.py \
--result_root /media/Data2/zhouyuemei/fulldata_ARefSR_results \
--model_name AttentionRefSR_pretrain_hr_syn_guided_ASPP_lowres_image_psv_plus_fuse_net \
--data_split test_shuffle2 \
--output_table /media/Data2/zhouyuemei/fulldata_ARefSR_results/AttentionRefSR_pretrain_hr_syn_guided_ASPP_lowres_image_psv_plus_fuse_net.json