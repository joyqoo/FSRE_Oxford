import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.kitti_dataset import KittiDataset
from networks.cma import CMA
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from options import Options
from utils import readlines
from utils.depth_utils import disp_to_depth

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """計算預測和 Ground truth 深度之間的誤差度量
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    """ 使用指定的測試集評估預訓練模型
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "請選擇 mono or stereo 評估 利用指令選擇  --eval_mono 或 --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "無法找到檔案從 {}".format(opt.load_weights_folder)

        print("->  讀取權重...從 {}".format(opt.load_weights_folder)," 中取得資料")

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        dataset = KittiDataset(height=encoder_dict['height'], width=encoder_dict['width'],
                               frame_idxs=[0], filenames=filenames, data_path=opt.data_path, is_train=False,
                               num_scales=len(opt.scales))
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=2,
                                pin_memory=True, drop_last=False)
        encoder = ResnetEncoder(num_layers=opt.num_layers)

        if not opt.no_cma:
            depth_decoder = CMA(encoder.num_ch_enc, opt=opt)
            decoder_path = os.path.join(opt.load_weights_folder, "decoder.pth")
        else:
            depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=opt.scales, opt=opt)


        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        pred_disps = []
        models = {}

        models['encoder'] = encoder
        models['depth'] = depth_decoder

        print("-> 用來執行預測的尺寸為 {}x{}".format(encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                for key in data:
                    data[key] = data[key].cuda()
                input_color = data[("color", 0, 0)]
                features = models['encoder'](input_color)
                if not opt.no_cma:
                    output, _ = models['depth'](features)
                else:
                    output = models["depth"](features)
                pred_disp = output[("disp", 0)]
                pred_disp, _ = disp_to_depth(pred_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
    else:
        # Load predictions from file
        print("-> 讀取預測...從 {}".format(opt.ext_disp_to_eval)," 中取得資料")
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.no_eval:
        print("-> 評估無效...結束！")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> 儲存 benchmark 預測到 {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            # depth = STEREO_SCALE_FACTOR * 5.2229753 / disp_resized
            depth = 32.779243 / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> KITTI benchmark 沒有可用的Ground-truth, 因此不進行評估...結束！")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> 評估中...")

    if opt.eval_stereo:
        print("   Stereo 評估 - 中值縮放無效, 縮放比例為 {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono 評估 - 使用中值縮放")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" 縮放比例 | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        print(med, np.mean(ratios))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> 完成!")


if __name__ == "__main__":
    options = Options()
    evaluate(options.parse())
