import cv2
import numpy as np
import colour
import Img_Data_Process as IDP
from logging.config import dictConfig
import logging
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import math
from PIL import Image

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
            'filename': 'debug.log',
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

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def d_psnr(_mse):
    if _mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(_mse))

if __name__ == '__main__':
    # 데이터 경로 설정 -- User setting
    root_path: str = r"C:\DataSET\ImageData\High-resolution pattern checker"
    data_1_path: str = r"Step 2_GroundTruth(SR+CR)"
    data_2_path: str = r"Step 2_DL SR+CR-38.124ms"

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

    # 계산 위해 실수화
    print("==========================================")
    print("\n Float32")
    img_data_1_org = img_data_1
    img_data_2_org = img_data_2
    for make_float_number in tqdm(range(img_data_1.shape[0])):
        img_data_1[make_float_number][:, :, :] = cv2.cvtColor(
            img_data_1[make_float_number][:, :, :].astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    for make_float_number in tqdm(range(img_data_2.shape[0])):
        img_data_2[make_float_number][:, :, :] = cv2.cvtColor(
            img_data_2[make_float_number][:, :, :].astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
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
                "| MSE | " + str(img_mse) +
                "| PSNR | " + str(img_psnr) +
                "| SSMI | " + str(img_ssim) +
                "| delta_E | " + str(np.mean(delta_E)))

        C_number += 1

    logging.debug("Data number: " + str(
        (len(img_file_name_list_A) + len(img_file_name_list_B)) / 2) + ", Calculated data number: " + str(C_number))
    print("Data number: " + str(
        (len(img_file_name_list_A) + len(img_file_name_list_B)) / 2) + ", Calculated data number: " + str(C_number))
