import glob
import logging
import math
import multiprocessing
from logging.config import dictConfig
from multiprocessing import Process
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


def calculate_all(_data_1_path, _data_2_path):
    data_1_path_image = cv2.imread(_data_1_path)
    data_2_path_image = cv2.imread(_data_2_path)

    img_mse = mse(data_1_path_image, data_2_path_image)
    img_psnr = d_psnr(img_mse)
    img_ssim = ssim(data_1_path_image, data_2_path_image,
                    multichannel=True)
    delta_E = colour.delta_E(data_1_path_image.astype("float"),
                             data_2_path_image.astype("float"), method="CIE 2000")

    return img_mse, img_psnr, img_ssim, np.mean(delta_E)


def test_1(_task_number, _data_1_path_img_files_list, _data_2_path_img_files_list,
           _add_str, _reference_file_path, _return_dict):
    _print_data = []
    if _task_number == 1:
        for number, _data_1_path_img_file in enumerate(tqdm(_data_1_path_img_files_list)):
            try:
                _data_2_path_img_file = _data_2_path_img_files_list[number]
                # print(_data_1_path_img_file)
                # print(_data_2_path_img_file)
                data_1_file_name = Path(_data_1_path_img_file).stem
                data_2_file_name = Path(_data_2_path_img_file).stem
                c_data_1_file_name = data_1_file_name + _add_str

                if c_data_1_file_name == data_2_file_name:

                    img_mse, img_psnr, img_ssim, delta_E_mean = calculate_all(_data_1_path_img_file,
                                                                              _data_2_path_img_file)

                    _print_data.append(
                        "A: " + data_1_file_name +
                        ", B: " + data_2_file_name +
                        " | MSE | " + str(img_mse) +
                        " | PSNR | " + str(img_psnr) +
                        " | SSIM | " + str(img_ssim) +
                        " | Delta_E | " + str(delta_E_mean)
                    )

                else:
                    raise Exception('c_data_1_file_name =1 data_2_file_name')

            except Exception as e:
                print('try1 error. : ', e)
                for _data_2_path_img_file in _reference_file_path:
                    data_1_file_name = Path(_data_1_path_img_file).stem
                    data_2_file_name = Path(_data_2_path_img_file).stem
                    c_data_1_file_name = data_1_file_name + _add_str

                    if c_data_1_file_name == data_2_file_name:
                        img_mse, img_psnr, img_ssim, delta_E_mean = calculate_all(_data_1_path_img_file,
                                                                                  _data_2_path_img_file)

                        _print_data.append(
                            "A: " + data_1_file_name +
                            ", B: " + data_2_file_name +
                            " | MSE | " + str(img_mse) +
                            " | PSNR | " + str(img_psnr) +
                            " | SSIM | " + str(img_ssim) +
                            " | Delta_E | " + str(delta_E_mean)
                        )
                        break

    else:
        for number, _data_1_path_img_file in enumerate(_data_1_path_img_files_list):
            try:
                _data_2_path_img_file = _data_2_path_img_files_list[number]
                data_1_file_name = Path(_data_1_path_img_file).stem
                data_2_file_name = Path(_data_2_path_img_file).stem
                c_data_1_file_name = data_1_file_name + _add_str

                if c_data_1_file_name == data_2_file_name:

                    img_mse, img_psnr, img_ssim, delta_E_mean = calculate_all(_data_1_path_img_file,
                                                                              _data_2_path_img_file)

                    _print_data.append(
                        "A: " + data_1_file_name +
                        ", B: " + data_2_file_name +
                        " | MSE | " + str(img_mse) +
                        " | PSNR | " + str(img_psnr) +
                        " | SSIM | " + str(img_ssim) +
                        " | Delta_E | " + str(delta_E_mean)
                    )

                else:
                    raise Exception('c_data_1_file_name =1 data_2_file_name')

            except Exception as e:
                print('try1 error. : ', e)
                for _data_2_path_img_file in _reference_file_path:
                    data_1_file_name = Path(_data_1_path_img_file).stem
                    data_2_file_name = Path(_data_2_path_img_file).stem
                    c_data_1_file_name = data_1_file_name + _add_str

                    if c_data_1_file_name == data_2_file_name:
                        img_mse, img_psnr, img_ssim, delta_E_mean = calculate_all(_data_1_path_img_file,
                                                                                  _data_2_path_img_file)

                        _print_data.append(
                            "A: " + data_1_file_name +
                            ", B: " + data_2_file_name +
                            " | MSE | " + str(img_mse) +
                            " | PSNR | " + str(img_psnr) +
                            " | SSIM | " + str(img_ssim) +
                            " | Delta_E | " + str(delta_E_mean)
                        )
                        break

    _return_dict[_task_number] = _print_data


def main(_task_number, _data_1_path_img_files_list, _data_2_path_img_files_list,
         _add_str, _reference_file_path, _return_dict):
    print("Start Process: ", _task_number)
    try:
        test_1(_task_number, _data_1_path_img_files_list, _data_2_path_img_files_list,
               _add_str, _reference_file_path, _return_dict)

    except OSError as err:
        print("OS error: {0}".format(err))


if __name__ == '__main__':
    # 데이터 경로 설정 -- User setting
    data_1_path: str = r"C:\DataSET\High-resolution pattern checker\210730\210730_CR GroundTruth"
    data_2_path: str = r"C:\DataSET\High-resolution pattern checker\210730\210730_CR Result_175000"

    # 데이터 읽어오기
    data_1_path_img_files_list = glob.glob(data_1_path + '/*.jpg')
    data_2_path_img_files_list = glob.glob(data_2_path + '/*.jpg')

    add_str = ''

    number_of_Process = 5
    number_of_data = len(data_1_path_img_files_list)

    print(math.ceil(number_of_data / number_of_Process))
    number_of_task = math.ceil(number_of_data / number_of_Process)

    data_1_path_test_job_distribution = []
    data_2_path_test_job_distribution = []

    for task_number in range(1, number_of_Process + 1):
        if number_of_task == 1:
            data_1_path_test_job_distribution.append(
                [data_1_path_img_files_list[number_of_task * task_number - number_of_task]])
            data_2_path_test_job_distribution.append(
                [data_2_path_img_files_list[number_of_task * task_number - number_of_task]])

        elif number_of_task > 1:
            data_1_path_test_job_distribution.append(
                data_1_path_img_files_list[number_of_task * task_number - number_of_task: number_of_task * task_number])
            data_2_path_test_job_distribution.append(
                data_2_path_img_files_list[number_of_task * task_number - number_of_task: number_of_task * task_number])

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []

    for task_number in range(1, number_of_Process + 1):
        th0 = Process(target=main, args=(task_number, data_1_path_test_job_distribution[task_number-1], data_2_path_test_job_distribution[task_number-1], add_str, data_2_path_img_files_list, return_dict))
        jobs.append(th0)
        th0.start()

    for proc in jobs:
        proc.join()

    for data in return_dict.keys():
        print(data)

    for data in return_dict.values():
        for a in data:
            logging.debug(a)
