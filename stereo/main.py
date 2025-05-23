import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import sys
import os
# import open3d as o3d

class DepthMap:

    def __init__(self, img_size, baseline, fx, fy, cx, cy, fov, window_size=7, min_disp=0, num_disp=35):
        
        block_size = window_size

        self.img_size = img_size
        self.baseline = baseline
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov = fov

        self.K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
        R = np.eye(3)
        T = np.array([baseline, 0, 0])

        self.R1, self.R2, self.P1, self.P2, _, _, _ = cv.stereoRectify(
            self.K, None, self.K, None, img_size, R, T
        )

        self.stereo = cv.StereoSGBM.create(
            minDisparity=min_disp,
            numDisparities=16*num_disp-min_disp,
            blockSize=block_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def rectify(self):
        map1_x, map1_y = cv.initUndistortRectifyMap(self.K, None, self.R1, self.P1, self.img_size, cv.CV_32FC1)
        map2_x, map2_y = cv.initUndistortRectifyMap(self.K, None, self.R2, self.P2, self.img_size, cv.CV_32FC1)

        rectified_left = cv.remap(self.imgLeft, map1_x, map1_y, cv.INTER_LINEAR)
        rectified_right = cv.remap(self.imgRight, map2_x, map2_y, cv.INTER_LINEAR)
        
        self.imgLeft = rectified_left
        self.imgReft = rectified_right

    def disparity(self):
        disparity = self.stereo.compute(self.imgLeft, self.imgRight)    
        disparity[disparity == 0] = 1

        disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        self.disparityMapVis = disparity_normalized
        self.disparityMap = disparity

    def depth(self):
        self.disparity_actual = self.disparityMap.astype(np.float32) * (self.stereo.getNumDisparities() / 255.0)

        depth = (self.fx * self.baseline) / self.disparityMap

        self.depthMap = depth

        transform = np.array([[0,0,-1],
                              [-1,0,0],
                              [0,-1,0]])
        X, Y = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
        X = X.astype(np.float64) - (depth.shape[1] - 1) / 2
        Y = Y.astype(np.float64) - (depth.shape[0] - 1) / 2
        Z = np.ones_like(depth) * (depth.shape[1] / (2*np.tan(self.fov/2)))

        direction_unit_vector = np.stack([X, Y, Z], axis=0) / np.sqrt(X*X + Y*Y + Z*Z)
        point_camera_frame = direction_unit_vector * depth

        points_and_var = np.concatenate([point_camera_frame, self.varianceMap[np.newaxis, :, :]], axis=0)
        points_and_var_camera_frame_unraveled = np.reshape(points_and_var, (4,-1))
        points_camera_frame_unraveled = points_and_var_camera_frame_unraveled[:3, :]
        var_unraveled = points_and_var_camera_frame_unraveled[3, :]

        left_cam_robot_frame = np.array([0.28, 0.081, 0.131])
        self.points = (transform @ points_camera_frame_unraveled).T + left_cam_robot_frame
        self.variances = np.squeeze(var_unraveled)
    
    def variance(self):
        # disparity_actual = self.disparityMap.astype(np.float32) * (self.stereo.getNumDisparities() / 255.0)

        # disparity_actual[disparity_actual <= 0] = np.nan

        sigma_d = 0.6

        self.disparityMap[self.disparityMap == 0] = 0.00001

        sigma_Z = ((self.fx * self.baseline / self.disparityMap**2) * sigma_d)
        sigma_Z2 = sigma_Z**2

        sigma_Z2[0:700, :] = np.inf
        self.varianceMap = sigma_Z2

    def compute(self, imgLeft, imgRight, path):
        # self.imgLeft = cv.imread(imgLeft, cv.IMREAD_GRAYSCALE)
        # self.imgRight = cv.imread(imgRight, cv.IMREAD_GRAYSCALE)

        self.imgLeft = imgLeft
        self.imgRight = imgRight

        self.disparity()

        self.variance()

        self.depth()
        output_path = os.path.join("/home/krzyzehj/work/final_project/SPIdepth/stereo/output", os.path.basename(path) + "_output.png")
        cv.imwrite(output_path, self.disparityMapVis)

        
def read_calibration_file(filepath):
    calibration = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            # Handle camera matrix (e.g. cam0, cam1)
            if value.startswith('[') and value.endswith(']'):
                value = value[1:-1].replace(';', '\n')  # convert to row-separated string
                matrix = np.fromstring(value, sep=' ').reshape(-1, 3)
                calibration[key] = matrix
            else:
                try:
                    calibration[key] = float(value)
                except ValueError:
                    calibration[key] = value  # fallback in case of string

    return calibration
    
def main(args):
    folder_path = args[1]
    try:            
        calib_path = os.path.join(folder_path, "calib.txt")
        left_img_path = os.path.join(folder_path, "im0.png")
        right_img_path = os.path.join(folder_path, "im1.png")
        
        calib = read_calibration_file(calib_path)
        left_img = cv.imread(left_img_path, cv.IMREAD_UNCHANGED)
        right_img = cv.imread(right_img_path, cv.IMREAD_UNCHANGED)
        
        camL = calib['cam0']
        camR = calib['cam1']
        fx = camL[0][0]
        fy = camL[1][1]
        cx = camL[0][2]
        cy = camL[1][2]
        baseline = calib['baseline']
        width = int(calib['width'])
        height = int(calib['height'])
        fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
        
        stereo = DepthMap(img_size=(width, height), baseline=baseline, fx=fx, fy=fy, cx=cx, cy=cy, fov=fov_x)
        
        stereo.compute(left_img, right_img, folder_path)

    except FileNotFoundError:
        print("Folder path not found.")
    except NotADirectoryError:
        print("Folder is not a directory.")

if __name__ == "__main__":
    main(sys.argv)