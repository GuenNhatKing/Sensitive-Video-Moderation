import cv2
import logging

def get_video_info(video_path):
    """Trích xuất metadata của video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logging.error(f"Không thể mở được video tại đường dẫn: {video_path}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    
    info = {
        "fps": fps,
        "total_frames": frames,
        "duration": duration
    }
    
    logging.info(f"Video Info - FPS: {fps:.2f}, Frames: {frames}, Duration: {duration:.2f}s")
    return info