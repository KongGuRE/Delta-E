import datetime
import logging
import math
from logging.config import dictConfig

import colour
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import Img_Data_Process as IDP

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
    err = err / 3 #이미지가 3채널 임으로 /3을 넣어 주어야함

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def d_psnr(_mse):
    if _mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(_mse))


def d_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
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

    start = datetime.datetime.now()
    # 데이터 경로 설정 -- User setting
    root_path: str = r"C:\DataSET\High-resolution pattern checker\210730"
    data_1_path: str = r"210730_CR GroundTruth"
    data_2_path: str = r"210730_CR Result_175000"

    # 데이터 읽어오기
    IP_1 = IDP.Img_Data_Loader()
    Data_Path_List_1, img_file_name_list_A = IP_1.Load_Img_Path(root_path, data_1_path, ck=True)
    img_data_1 = IP_1.Load_Img_Data(Data_Path_List_1, ck=False)

    IP_2 = IDP.Img_Data_Loader()
    Data_Path_List_2, img_file_name_list_B = IP_2.Load_Img_Path(root_path, data_2_path, ck=True)
    img_data_2 = IP_2.Load_Img_Data(Data_Path_List_2, ck=False)

    # 읽어온 데이터 계산전 이상여부 검증
    print("==========================================")
    print("Pre-Calculation Data Validation")

    if len(img_file_name_list_A) != len(img_file_name_list_B):
        print("Check data length : Fail")
        logging.critical("Data A and B do not have the same number. Check the data.")
    else:
        print("Check data length : Ok")

    number = 0
    for data_maching_A in tqdm(img_file_name_list_A):
        for data_maching_B in img_file_name_list_B:
            if data_maching_A == data_maching_B:
                number += 1
                break

    if (len(img_file_name_list_A) + len(img_file_name_list_B)) / 2 == number:
        print("Check data matching : Ok")
    else:
        print("Check data matching : Fail")
        logging.warning("There is an object in the image whose file name does not match. This can cause "
                        "problems with the results.")
    print("==========================================")

    # delta_E 계산
    C_number = 0
    print("\ndelta E calculating")
    for calculate_delta_e_number, file_name in enumerate(tqdm(img_file_name_list_A)):
        try:
            B_index = img_file_name_list_B.index(file_name)
        except:
            continue
        if img_file_name_list_B[B_index] == file_name:
            img_mse = mse(img_data_1[calculate_delta_e_number][:, :, :], img_data_2[B_index][:, :, :])
            img_psnr = d_psnr(img_mse)
            img_ssim = ssim(img_data_1[calculate_delta_e_number][:, :, :], img_data_2[B_index][:, :, :],
                            multichannel=True)
            delta_E = colour.delta_E(img_data_1[calculate_delta_e_number][:, :, :].astype("float"),
                                     img_data_2[B_index][:, :, :].astype("float"), method="CIE 2000")

            logging.debug(
                "A: " + file_name + ", B: " + img_file_name_list_B[B_index] +
                " | MSE | " + str(img_mse) +
                " | PSNR | " + str(img_psnr) +
                " | SSMI | " + str(img_ssim) +
                " | Delta_E | " + str(np.mean(delta_E)))

        C_number += 1

    # logging.debug("Data number: " + str(
    #     (len(img_file_name_list_A) + len(img_file_name_list_B)) / 2) + ", Calculated data number: " + str(C_number))
    print("Data number: " + str(
        (len(img_file_name_list_A) + len(img_file_name_list_B)) / 2) + ", Calculated data number: " + str(C_number))
