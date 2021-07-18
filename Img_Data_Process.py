import os
import cv2
import numpy as np
from tqdm import tqdm

class Img_Data_Loader:
    def __init__(self):
        self.Img_Data_Names = []
        self.Img_Data_file_name = []

    def Load_Img_Path(self, root_dir: str, data_path: str, ck: bool = False, ck_file_number: bool = False):
        for (root, dirs, files) in os.walk(root_dir):
            if root == os.path.join(root_dir, data_path):
                if ck:
                    print(os.path.join(root))
                if len(files) > 0:
                    for file_name in files:
                        self.Img_Data_Names.append(os.path.join(root, file_name))
                        self.Img_Data_file_name.append(file_name)
                        if ck_file_number:
                            print(os.path.join(root, file_name))

        return self.Img_Data_Names, self.Img_Data_file_name

    def Load_Img_Data_convert_RGB(self, file_names: list, ck: bool = False, ck_file_number: bool = False):
        for File_idx, imgFile in enumerate(file_names):
            colored_img = cv2.imread(imgFile)
            RGBimg = np.zeros((len(file_names), colored_img.shape[2], colored_img.shape[1], colored_img.shape[0]))
            break
        if ck_file_number:
            print("==========================================")
            print("RGBimg : " + str(RGBimg.shape))
            print("==========================================")

        for File_idx, imgFile in enumerate(file_names):
            if ck:
                print("File_idx: ", File_idx)
            coloredImg = cv2.imread(imgFile)
            b, g, r = cv2.split(coloredImg)
            RGBimg[File_idx, 0, :, :] = r
            RGBimg[File_idx, 1, :, :] = g
            RGBimg[File_idx, 2, :, :] = b
        if ck:
            print("==========================================")
            print("Data shape : " + str(RGBimg.shape))
            print("==========================================")
        return RGBimg

    def Load_Img_Data(self, file_names: list, ck: bool = False, ck_file_number: bool = False):

        img_data: list = []

        if ck:
            for File_idx, imgFile in enumerate(tqdm(file_names)):
                if ck_file_number:
                    print("File_idx: ", File_idx)
                coloredImg = cv2.imread(imgFile)
                img_data.append(coloredImg)
            img_data = np.array(img_data)

            print("==========================================")
            print("Data shape : " + str(img_data.shape))
            print("==========================================")
        else:
            for File_idx, imgFile in enumerate(file_names):
                if ck_file_number:
                    print("File_idx: ", File_idx)
                coloredImg = cv2.imread(imgFile)
                img_data.append(coloredImg)
            img_data = np.array(img_data)
        return img_data


if __name__ == '__main__':
    Img_data = Img_Data_Loader()
    Data_Path_List, _ = Img_data.Load_Img_Path(r"C:\DataSET\ImegeData\HTCC_Crack\HTCC_Crack\Test", r"OK", ck=True)
    Img_data = Img_data.Load_Img_Data(Data_Path_List, ck=True)

    print(Img_data.shape)

    image1_lab = cv2.cvtColor(Img_data[0, :, :, :].astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    image2_lab = cv2.cvtColor(Img_data[1, :, :, :].astype(np.float32) / 255, cv2.COLOR_RGB2Lab)

    print(image1_lab.shape)
    print(image2_lab.shape)
    import colour

    delta_E = colour.delta_E(image1_lab, image2_lab, method="CIE 2000")
    print(np.mean(delta_E))































"""
██╗░░██╗░█████╗░███╗░░██╗░██████╗░░██████╗░██╗░░░██╗██████╗░███████╗
██║░██╔╝██╔══██╗████╗░██║██╔════╝░██╔════╝░██║░░░██║██╔══██╗██╔════╝
█████═╝░██║░░██║██╔██╗██║██║░░██╗░██║░░██╗░██║░░░██║██████╔╝█████╗░░
██╔═██╗░██║░░██║██║╚████║██║░░╚██╗██║░░╚██╗██║░░░██║██╔══██╗██╔══╝░░
██║░╚██╗╚█████╔╝██║░╚███║╚██████╔╝╚██████╔╝╚██████╔╝██║░░██║███████╗
╚═╝░░╚═╝░╚════╝░╚═╝░░╚══╝░╚═════╝░░╚═════╝░░╚═════╝░╚═╝░░╚═╝╚══════╝
"""