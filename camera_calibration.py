import cv2 as cv
import numpy as np

video_path = 'chessboard.mp4'   # 파일 경로
chessboard_size = (10, 7)       # 체스보드 내 코너 개수 (가로, 세로)
square_size = 2.5               # 각 정사각형 실제 크기

def detect_chessboard_corners(video_path, chessboard_size, square_size, display_delay=500, max_frames=30, frame_interval=0.5):
    """
    Args:
        display_delay: 코너 검출 결과 보여줄 때의 지연 시간 (ms)
        max_frames: 캘리브레이션에 사용할 최대 프레임 수
        frame_interval: 이전에 검출된 프레임으로부터 수집할 최소 시간 간격 (초)
    
    Return:
        objpoints : 실제 3D 객체 포인트 리스트
        imgpoints : 영상상의 2D 코너 포인트 리스트
        image_size: 영상의 크기 (width, height)
    """
    
    # 3D 객체 포인트 준비: (0,0,0), (1,0,0), (2,0,0), ...
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 캘리브레이션에 사용할 모든 3D 객체 포인트와 2D 이미지 포인트를 저장하는 리스트
    objpoints = []  # 실제 3D 포인트
    imgpoints = []  # 영상 2D 포인트
    
    # 코너 서브픽셀 보정 위한 종료 기준
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    cap = cv.VideoCapture(video_path)
    frame_count = 0
    detected_frames = 0
    image_size = None
    
    # 마지막 프레임 수집 시각 (초 단위)
    last_capture_time = -frame_interval
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        frame_count += 1
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
            
        # 현재 프레임의 시간(초 단위)을 계산
        time_stamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
        
        # 설정한 시간 간격보다 짧다면 넘어감
        if time_stamp - last_capture_time < frame_interval:
            continue
        
        # 시간 간격 만족 시 업데이트
        last_capture_time = time_stamp

        # 체스보드 코너 검출
        found, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if found:
            # 코너 위치 정밀화
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 검출된 코너 그리기 (디버깅 및 시각화용)
            cv.drawChessboardCorners(frame, chessboard_size, corners_refined, found)
            cv.putText(frame, f"Frame {frame_count}: Chessboard detected", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 객체 포인트와 이미지 포인트 저장
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            detected_frames += 1

            # 검출 결과 잠시 표시 (500ms)
            cv.imshow('Chessboard Detection', frame)
            cv.waitKey(display_delay)
        else:
            cv.imshow('Chessboard Detection', frame)
            # ESC 키를 누르면 조기 종료
            if cv.waitKey(30) & 0xFF == 27:
                break
        
        # 최대 수집 프레임 제한
        if detected_frames >= max_frames: break

    cap.release()
    cv.destroyAllWindows()
    
    return objpoints, imgpoints, image_size

def calibrate_camera(objpoints, imgpoints, image_size):
    """
    주어진 객체 및 이미지 포인트를 사용하여 카메라 캘리브레이션
    
    Return:
      ret         : RMS 재투영 오차
      camera_matrix: 캘리브레이션된 카메라 행렬 (K)
      dist_coeffs : 왜곡 계수
      rvecs       : 각 프레임별 회전 벡터
      tvecs       : 각 프레임별 평행 이동 벡터
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    
    print("캘리브레이션 결과:")
    print("RMS 재투영 오차:", ret)
    print("카메라 행렬 (K):\n", camera_matrix)
    print("왜곡 계수:\n", dist_coeffs)
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

# Step 3: 왜곡 보정 시연
def undistort_video(video_path, camera_matrix, dist_coeffs):
    cap = cv.VideoCapture(video_path)
    
    fps = cap.get(cv.CAP_PROP_FPS)
    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    
    out = cv.VideoWriter("demo.avi", fourcc, fps, (orig_width, orig_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
        
        out.write(undistorted)
        
        combined = np.hstack((frame, undistorted))
        cv.imshow("Original (좌측) vs Undistorted (우측)", combined)
        
        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

# 실행
if __name__ == '__main__':
    # 1. 코너 검출
    objpoints, imgpoints, image_size = detect_chessboard_corners(video_path, chessboard_size, square_size)
    
    # 2. 카메라 캘리브레이션
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)
    
    # 3. 렌즈 왜곡 보정 영상 시연
    undistort_video(video_path, camera_matrix, dist_coeffs)