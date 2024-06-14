import cv2
import numpy as np
import glob
import os

# 체스보드 크기 정의
chessboard_size = (9, 6)
square_size = 1.0  # 체스보드의 각 사각형 한 변의 길이 (실제 크기, 단위: cm)

# 체스보드의 3D 좌표 준비
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D 좌표를 저장할 리스트
imgpoints = []  # 2D 좌표를 저장할 리스트

# 캘리브레이션 이미지 로드
images = glob.glob('*.png')

if not images:
    raise FileNotFoundError("No calibration images found in the specified folder.")

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너를 그려서 확인
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Image', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard corners not found in image {os.path.basename(image_file)}")

cv2.destroyAllWindows()

if not objpoints or not imgpoints:
    raise ValueError("Chessboard corners not found in any of the images. Calibration failed.")

# 카메라 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 캘리브레이션 결과 저장
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs)
