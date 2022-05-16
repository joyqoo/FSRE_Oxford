
from copy import deepcopy
from PIL import Image
from torchvision import transforms

import os
import cv2
import random
import torch
import skimage.transform

import PIL.Image as pil
import numpy as np
import torch.utils.data as data


from utils.seg_utils import labels

# crop size
#_CROP_SIZE = (1152, 640)
# _CROP_SIZE = (640, 192)
_CROP_SIZE = (512, 256)
# half size
_HALF_SIZE = (576, 320)
# nuscenes size
_NUSCENES_SIZE = (768, 384)

# 載入PIL 影像轉成RGB 影像
def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    if mode == 'P':
        return Image.open(path)
    else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
      

class RobotcarDataset(data.Dataset):
    """
    Oxford RobotCar data set.
    """

    def __init__(self,
                 height,
                 width,
                 frame_idxs,
                 filenames,
                 data_path,
                 num_scales,
                 is_train,
                 img_ext='.png',
                 ):
        super(RobotcarDataset, self).__init__()

        """Superclass for monocular dataloaders

        Args:
            data_path: 數據路徑
            filenames: 文件名
            height: 高度
            width: 寬度
            frame_idxs: 幀索引
            num_scales: 比例數
            is_train: 是否為訓練
            img_ext: 是否增加
        """

        # 設置參數
        self.data_path = data_path
        self.filenames = filenames
        # self.height = height
        # self.width = width
        self.width, self.height = _CROP_SIZE
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.num_scales = num_scales


        try:
            # 亮度
            self.brightness = (0.8, 1.2)
            # 對比
            self.contrast = (0.8, 1.2)
            # 飽和
            self.saturation = (0.8, 1.2)
            # 色調
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # 讀取內在函數
        self.K = self.load_intrinsic()
        
        # 調整影像和分割圖尺寸
        self.resize_img = {}
        self.resize_seg = {}
        for i in range(self.num_scales):
            # 1、2、4、8、16、32 遞增
            s = 2 ** i
            self.resize_img[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.ANTIALIAS)
            self.resize_seg[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.BILINEAR)
             
        # self.load_depth = False

    # 影像預處理
    def preprocess(self, inputs, color_aug):
        """將RGB 影像調整到需要的比例並在需要時擴充
        
        預先建立 color_aug ，並且把一樣的增強應用於同項目的所有影像
        確保了所有輸入到Pose seg_networks 的影像都有相同的增強。
        """
        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_img[i](inputs[(n, im, -1)])
                # 移除整個列表
                del inputs[(n, im, -1)]

            if "seg" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_seg[i](inputs[(n, im, -1)])
                # 移除整個列表
                del inputs[(n, im, -1)]

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "seg" in k:
                n, im, i = k
                inputs[(n, im, i)] = torch.tensor(np.array(f)).float().unsqueeze(0)
        
    # 返回長度或個數
    def __len__(self):
        return len(self.filenames)

    # 利用index 返回項目
    def __getitem__(self, index):

        """將資料集中的單個訓練項作為字典返回。

        Value 對應於tensor ，字典中的Key 是字符串或元組：
            ("color", <frame_id>, <scale>) 用於原始彩色影像，
            ("color_aug", <frame_id>, <scale>) 用於增強的彩色影像，
            ("K", scale) 或 ("inv_K", scale) 用於相機內在函數，
            “stereo_T”用於相機外部，以及Ground-truth 深度圖的“depth_gt”

        <frame_id> 是：
            表示相對於“index”的時間步長的整數（例如 0 或-1 或 1），
        或者
            “s”表示立體對中的相反影像。

        <scale> 是一個整數，表示影像相對於全尺寸影像的比例：
            -1 從硬碟加載的原始分辨率影像
            0 張影像大小調整為 (self.width, self.height)
            1 張影像大小調整為 (self.width // 2, self.height // 2)
            2 張影像大小調整為 (self.width // 4, self.height // 4)
            3 張影像大小調整為 (self.width // 8, self.height // 8)

        """
        inputs = {}
        # 隨機讓訓練數據進行顏色增強的預處理
        do_color_aug = self.is_train and random.random() > 0.5
        # 隨機讓訓練數據進行水平左右翻轉的預處理
        do_flip = self.is_train and random.random() > 0.5
        # index 是train_txt 中的第index 行
        line = self.filenames[index].split()

        # train_files.txt 中一行數據的第二部分，即影像位置
        folder = int(line[1])

        if len(line) in [2]:
            # train_files.txt 中一行數據的第一部分，即影像名稱
            frame_index = int(line[0])
        else:
            frame_index = 0


        for i in self.frame_idxs:
            if frame_index == 22200 or frame_index == 1: 
        
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, do_flip)
                
            # elif frame_index == 7369:
            #     inputs[("color", i, -1)] = self.get_color(folder, frame_index, do_flip) 

            else:
                try:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, do_flip)
                except Exception as e:
                    try:
                        inputs[("color", i, -1)] = self.get_color(folder + i , frame_index + i, do_flip)
                    except Exception as e:
                        try:
                            inputs[("color", i, -1)] = self.get_color(folder - i , frame_index + i, do_flip)
                        except Exception as e:
                            print('error message: ',e)
                            pass

        inputs[("seg", 0, -1)] = self.get_seg_map(folder, frame_index, do_flip)
       
        K = deepcopy(self.K)
        if do_flip:
            K[0, 2] = 1 - K[0, 2]

        K[0, :3] *= self.width
        K[1, :3] *= self.height
        inv_K = np.linalg.pinv(K)
        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        # 取得 Ground-truth 深度
        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, do_flip)
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs


    # 載入內在函數
    def load_intrinsic(self):

        intrinsic = np.array([
            [400.0, 0.0, 500.107605, 0.0],
            [0.0, 400.0, 511.461426, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return intrinsic

    # 取得顏色
    def get_color(self, folder, frame_index, do_flip):
        im_path = self.get_image_path(folder, frame_index)
        color = self.loader(im_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    # 取得影像路徑
    def get_image_path(self, folder, frame_index):

        fill_frame_index = str(frame_index).zfill(6)
        f_str = fill_frame_index + self.img_ext
        next_folder = '2014-12-16-18-44-24/stereo/left'

        image_path = os.path.join(
            self.data_path, "2014-12-16-18-44-24_stereo_left_0{}/".format(folder), next_folder, f_str)

        return image_path

    # 取得真實深度(未實做)
    def get_depth(self, folder, frame_index, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


    # 取得分割圖
    def get_seg_map(self, folder, frame_index, do_flip):
        path = self.get_image_path(folder, frame_index)
        # path = path.replace('Robotcar', 'Robotcar/segmentation')
        path = path.replace('Robotcar_full', 'Robotcar_full/segmentation')
        seg = self.loader(path, mode='P')
        seg_copy = np.array(seg.copy())

        # for k in np.unique(seg):
        #     seg_copy[seg_copy == k] = labels[k].trainId
        seg = Image.fromarray(seg_copy, mode='P')

        if do_flip:
            seg = seg.transpose(pil.FLIP_LEFT_RIGHT)
        return seg
