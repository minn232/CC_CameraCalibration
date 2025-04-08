import cv2 as cv
import numpy as np

video_path = 'chessboard.mp4'
board_pattern = (10, 7)
board_cellsize = 2.5
wait_msec = 1000

def select_img_from_video(video_file, board_pattern, max_frames=20):
    # Open a video
    video = cv.VideoCapture(video_file)
    selected_imgs = []
    
    while len(selected_imgs) < max_frames:
        ret, frame = video.read()
        if not ret: break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, board_pattern)
        if found:
            cv.drawChessboardCorners(frame, board_pattern, corners, found)
            selected_imgs.append(frame.copy())

        cv.imshow('Frame Selection', frame)
        if cv.waitKey(wait_msec) == 27:
            break
    
    video.release()
    cv.destroyAllWindows()
    return selected_imgs

def calibrate_camera(images, board_pattern, board_cellsize):
    objp = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp *= board_cellsize

    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, board_pattern)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return ret, K, dist, rvecs, tvecs

# Step 3: 왜곡 보정 시연
def demo_undistortion(video_path, K, dist):
    cap = cv.VideoCapture(video_path)
    map1, map2 = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if map1 is None:
            h, w = frame.shape[:2]
            map1, map2 = cv.initUndistortRectifyMap(K, dist, None, K, (w, h), cv.CV_32FC1)

        rectified = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)
        combined = np.hstack((frame, rectified))
        cv.imshow('Original (left) vs Rectified (right)', combined)

        if cv.waitKey(30) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

# 실행
if __name__ == '__main__':
    images = select_img_from_video(video_path, board_pattern)
    ret, K, dist, rvecs, tvecs = calibrate_camera(images, board_pattern, board_cellsize)

    print('--- Calibration Result ---')
    print(f'RMSE (re-projection error): {ret:.6f}')
    print(f'Camera Matrix (K):\n{K}')
    print(f'Distortion Coefficients:\n{dist.ravel()}')

    demo_undistortion(video_path, K, dist)