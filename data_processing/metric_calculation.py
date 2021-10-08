import os
import cv2
import numpy as np
import glob
import argparse
from skimage.measure import compare_psnr, compare_ssim

parser = argparse.ArgumentParser(description='Metric Calculation')

# data path
parser.add_argument('--data_split_path', type=str, default=None, help='The directory of data split.')
parser.add_argument('--pred_glob', type=str, default='ref_image*.png', help='The directory of data split.')
parser.add_argument('--gt_glob', type=str, default='tgt_image*.png', help='The directory of data split.')

args=parser.parse_args()

if __name__ == '__main__':
    assert os.path.isdir(args.data_split_path), '--data_split_path is not a directory, pls retry.'
    print('pred_glob = %s' % args.pred_glob)
    print('gt_glob = %s' % args.gt_glob)
    img_num = 0
    psnr_sum = 0
    ssim_sum = 0
    total_files_num = len(os.listdir(args.data_split_path))
    for dir_ in os.listdir(args.data_split_path):
        if not os.path.isdir(os.path.join(args.data_split_path, dir_)):
            continue
        pred_image = cv2.imread(glob.glob(os.path.join(args.data_split_path, dir_, args.pred_glob))[0])
        gt_image = cv2.imread(glob.glob(os.path.join(args.data_split_path, dir_, args.gt_glob))[0])
        psnr = compare_psnr(gt_image, pred_image)
        ssim = compare_ssim(gt_image, pred_image, multichannel=True)
        if psnr > 1000:
            print('psnr = %f' % psnr)
            print('It is folder %s' % dir_)
            continue
        if img_num % 1000 == 0:
            print('img_num:%05d/%d, psnr = %f, ssim = %f' % (img_num, total_files_num, psnr, ssim))
        psnr_sum = psnr_sum + psnr
        ssim_sum = ssim_sum + ssim
        img_num = img_num + 1

    psnr_avg = psnr_sum / img_num
    ssim_avg = ssim_sum / img_num
    print('total pairs = %d' % img_num)
    print('psnr_sum = %f, ssim_sum = %f' % (psnr_sum, ssim_sum))
    print('psnr_avg = %f, ssim_avg = %f' % (psnr_avg, ssim_avg))