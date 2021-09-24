import glob
import logging
import math
from logging.config import dictConfig
from pathlib import Path

import colour
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(message)s',
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'Calculat.log',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file']
    }
})


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err = err / 3  # 이미지가 3채널 임으로 /3을 넣어 주어야함

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def d_psnr(_mse):
    if _mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(_mse))


def d_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    # 데이터 경로 설정 -- User setting
    data_1_path: str = r"C:\DataSET\High-resolution pattern checker\210730\210730_CR GroundTruth"
    data_2_path: str = r"C:\DataSET\High-resolution pattern checker\210730\210730_CR Result_175000"

    # 데이터 읽어오기
    data_1_path_img_files = glob.glob(data_1_path + '/*.jpg')
    data_2_path_img_files = glob.glob(data_2_path + '/*.jpg')

    add_str = ''

    # delta_E 계산
    C_number = 0
    print("\ndelta E calculating")

    for number, data_1_path_Image_name in enumerate(tqdm(data_1_path_img_files)):
        try:
            data_2_path_Image_name = data_2_path_img_files[number]
            data_1_file_name = Path(data_1_path_Image_name).stem
            data_2_file_name = Path(data_2_path_Image_name).stem
            c_data_1_file_name = data_1_file_name + add_str

            if c_data_1_file_name == data_2_file_name:

                data_1_path_image = cv2.imread(data_1_path_Image_name)
                data_2_path_image = cv2.imread(data_2_path_Image_name)

                img_mse = mse(data_1_path_image, data_2_path_image)
                img_psnr = d_psnr(img_mse)
                img_ssim = ssim(data_1_path_image, data_2_path_image,
                                multichannel=True)
                delta_E = colour.delta_E(data_1_path_image.astype("float"),
                                         data_2_path_image.astype("float"), method="CIE 2000")

                logging.debug(
                    "A: " + data_1_file_name +
                    ", B: " + data_2_file_name +
                    " | MSE | " + str(img_mse) +
                    " | PSNR | " + str(img_psnr) +
                    " | SSMI | " + str(img_ssim) +
                    " | Delta_E | " + str(np.mean(delta_E)))
            else:
                raise Exception('c_data_1_file_name =1 data_2_file_name')

        except Exception as e:
            print('try1 error. : ', e)
            for data_2_path_Image_name in data_2_path_img_files:
                data_1_file_name = Path(data_1_path_Image_name).stem
                data_2_file_name = Path(data_2_path_Image_name).stem
                c_data_1_file_name = data_1_file_name + add_str

                if c_data_1_file_name == data_2_file_name:
                    data_1_path_image = cv2.imread(data_1_path_Image_name)
                    data_2_path_image = cv2.imread(data_2_path_Image_name)

                    img_mse = mse(data_1_path_image, data_2_path_image)
                    img_psnr = d_psnr(img_mse)
                    img_ssim = ssim(data_1_path_image, data_2_path_image,
                                    multichannel=True)
                    delta_E = colour.delta_E(data_1_path_image.astype("float"),
                                             data_2_path_image.astype("float"), method="CIE 2000")

                    logging.debug(
                        "A: " + data_1_file_name +
                        ", B: " + data_2_file_name +
                        " | MSE | " + str(img_mse) +
                        " | PSNR | " + str(img_psnr) +
                        " | SSMI | " + str(img_ssim) +
                        " | Delta_E | " + str(np.mean(delta_E)))

                    break

            C_number += 1

    # logging.debug("Data number: " + str(
    #     (len(img_file_name_list_A) + len(img_file_name_list_B)) / 2) + ", Calculated data number: " + str(C_number))
    print("Data number: " + str(
        (len(data_1_path_img_files) + len(data_2_path_img_files)) / 2) + ", Calculated data number: " + str(C_number))
