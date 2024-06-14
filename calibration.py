import cv2
import numpy as np
import glob

# 체스보드 크기 정의
chessboard_size = (9, 6)
frame_size = (640, 480)

# 체스보드의 3D 좌표 준비
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D 좌표를 저장할 리스트
imgpoints = []  # 2D 좌표를 저장할 리스트

# 캘리브레이션 이미지 로드
images = glob.glob('cali.webp')
if not images:
    raise FileNotFoundError("No calibration images found in the specified folder.")

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너를 그려서 확인
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Image', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

# 캘리브레이션 결과 저장
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs)
