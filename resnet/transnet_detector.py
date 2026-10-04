import sys
import cv2
import numpy as np
import torch
import logging
import json
from pathlib import Path

class SceneDetector:
    def __init__(self, transnet_dir, weights_path, device=None, threshold=0.5):
        """
        Khởi tạo SceneDetector.
        transnet_dir: Đường dẫn đến thư mục chứa mã nguồn TransNetV2 (để import).
        weights_path: Đường dẫn đến file trọng số .pth.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.window_size = 100
        self.stride = 50
        self.input_height = 27
        self.input_width = 48
        
        # Hỗ trợ nạp TransNetV2 từ thư mục local clone nếu có
        if transnet_dir:
            transnet_path = Path(transnet_dir).resolve()
            for p in [transnet_path, transnet_path / "inference-pytorch"]:
                p_str = str(p)
                if p.exists() and p_str not in sys.path:
                    sys.path.insert(0, p_str)
            
        try:
            from transnetv2_pytorch import TransNetV2
        except ImportError:
            raise ImportError(
                f"Không thể import TransNetV2. Vui lòng cài đặt 'transnetv2-pytorch' (pip install transnetv2-pytorch) "
                f"hoặc clone TransNetV2 từ GitHub theo hướng dẫn trong README.md (Đường dẫn hiện tại: {transnet_dir})."
            )

        self.model = self._load_model(TransNetV2, weights_path)

    def _load_model(self, model_class, weights_path):
        logging.info("Đang tải mô hình TransNetV2...")
        model = model_class()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")
            
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        logging.info(f"Tải mô hình TransNetV2 thành công trên thiết bị: {self.device}")
        return model

    def _read_video(self, video_path):
        logging.info(f"Đọc video để cắt cảnh: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            cap.release()
            raise RuntimeError("Lỗi FPS không hợp lệ.")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            
        cap.release()
        if not frames:
            raise RuntimeError("Không đọc được frame nào từ video.")

        frames = np.asarray(frames, dtype=np.uint8)
        logging.info(f"Đã tải {len(frames)} frames. Kích thước đầu vào: {frames.shape}")
        return frames, fps

    def _predict_frames(self, frames):
        logging.info("Bắt đầu chạy TransNetV2 inference...")
        no_padded_frames_start = 25
        remainder = len(frames) % 50
        no_padded_frames_end = 25 + 50 - remainder if remainder != 0 else 25

        start_frame = frames[0:1]
        end_frame = frames[-1:]

        padded_frames = np.concatenate([
            np.repeat(start_frame, no_padded_frames_start, axis=0),
            frames,
            np.repeat(end_frame, no_padded_frames_end, axis=0)
        ], axis=0)

        predictions = []
        ptr = 0
        total_frames = len(frames)

        with torch.no_grad():
            while ptr + self.window_size <= len(padded_frames):
                window = padded_frames[ptr:ptr + self.window_size]
                input_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
                
                single_frame_pred, _ = self.model(input_tensor)
                single_frame_pred = torch.sigmoid(single_frame_pred)
                pred = single_frame_pred[0, 25:75, 0].cpu().numpy()
                predictions.append(pred)
                ptr += self.stride

        predictions = np.concatenate(predictions)[:len(frames)]
        return predictions

    def _predictions_to_scenes(self, predictions):
        predictions_bin = (predictions > self.threshold).astype(np.uint8)
        scenes = []
        t_prev = 0
        start = 0

        for i, t in enumerate(predictions_bin):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if 't' in locals() and t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def detect_scenes(self, video_path, output_json_path=None):
        """
        Hàm chính để thực thi cắt cảnh.
        Trả về danh sách các cảnh (shots) dưới dạng dictionary.
        """
        frames, fps = self._read_video(video_path)
        predictions = self._predict_frames(frames)
        scenes = self._predictions_to_scenes(predictions)
        
        logging.info(f"Phát hiện được {len(scenes)} phân đoạn (shots).")
        
        shots_info = []
        for index, (start_frame, end_frame) in enumerate(scenes, start=1):
            start_frame, end_frame = int(start_frame), int(end_frame)
            start_time = start_frame / fps
            end_time = (end_frame + 1) / fps
            duration = (end_frame - start_frame + 1) / fps
            
            shots_info.append({
                "shot_id": index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": round(start_time, 3),
                "end_time_sec": round(end_time, 3),
                "duration_sec": round(duration, 3)
            })

        # Lưu log json
        if output_json_path:
            result = {
                "video": str(video_path),
                "fps": fps,
                "frame_count": len(frames),
                "duration_sec": round(len(frames) / fps, 3),
                "threshold": self.threshold,
                "shot_count": len(shots_info),
                "shots": shots_info
            }
            output_json_path = Path(output_json_path)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logging.info(f"Đã lưu kết quả phân cảnh vào: {output_json_path}")

        return shots_info, fps, len(frames)