import json
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Đưa thư mục src vào sys.path để import module dùng chung
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_config import get_env, resolve_video_path

def cut_adult_regions(json_path, output_dir=None, override_video_path=None, padding_sec=1.0, max_gap_sec=1.5):
    """
    Đọc JSON của Baseline, gom cụm các frame Adult gần nhau thành Region và cắt video.
    Tự động xác định đường dẫn video từ JSON mà không cần truyền --video.
    """
    logging.info(f"Đọc file JSON Baseline: {json_path}")
    
    if not os.path.exists(json_path):
        logging.error(f"Không tìm thấy file JSON: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy đường dẫn video từ JSON và tự động giải quyết vị trí file
    raw_video = override_video_path if override_video_path else data.get("video")
    video_path = resolve_video_path(raw_video, json_path=json_path)
    
    if not video_path or not os.path.exists(video_path):
        logging.error(f"Không tìm thấy file video gốc tại: {raw_video}")
        logging.info("Gợi ý: Dùng tham số --video để trỏ lại đường dẫn đúng nếu file đã bị di chuyển.")
        return

    logging.info(f"Đã xác định vị trí video gốc: {video_path}")

    frame_results = data.get("frame_results", [])
    
    # Lọc các frame được nhãn là adult
    adult_frames = [f for f in frame_results if f.get("prediction") == "adult"]
    
    if not adult_frames:
        logging.info("Video không có frame Adult nào để cắt.")
        return

    # Sắp xếp theo thời gian (cho chắc chắn)
    adult_frames.sort(key=lambda x: x["time_sec"])

    # THUẬT TOÁN GOM CỤM (GROUPING)
    regions = []
    current_region = [adult_frames[0]]

    for frame in adult_frames[1:]:
        prev_frame = current_region[-1]
        gap = frame["time_sec"] - prev_frame["time_sec"]
        
        if gap <= max_gap_sec:
            # Khung hình gần nhau -> Nhét chung vào 1 cảnh
            current_region.append(frame)
        else:
            # Khoảng cách quá xa -> Cắt cảnh mới
            regions.append(current_region)
            current_region = [frame]
    regions.append(current_region)

    logging.info(f"Đã gom {len(adult_frames)} frame thành {len(regions)} vùng liên tục.")

    # Cấu hình thư mục đầu ra
    if not output_dir:
        output_dir = get_env("BASELINE_CLIPS_OUT_DIR", "baseline_bad_clips", resolve=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    for idx, reg in enumerate(regions, start=1):
        start_sec = reg[0]["time_sec"]
        end_sec = reg[-1]["time_sec"]

        # Nếu vùng chỉ có 1 frame lướt qua (thời gian start = end), mở rộng nhẹ để ffmpeg không bị lỗi
        if start_sec == end_sec:
            end_sec += 0.5 
            
        # Thêm Padding bao quát (Mặc định mở rộng 1 giây 2 đầu)
        final_start = max(0, start_sec - padding_sec)
        final_end = end_sec + padding_sec

        out_filename = f"{video_name}_baseline_reg{idx:02d}_{final_start:.1f}s_to_{final_end:.1f}s.mp4"
        out_filepath = os.path.join(output_dir, out_filename)

        cmd = [
            "ffmpeg", "-y", 
            "-ss", str(final_start), 
            "-to", str(final_end), 
            "-i", str(video_path), 
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", 
            out_filepath
        ]

        logging.info(f"Đang cắt Region {idx:02d} (Gồm {len(reg)} frames, {final_start:.2f}s -> {final_end:.2f}s)...")
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            logging.info(f"-> Đã lưu: {out_filepath}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Lỗi khi cắt Region {idx}: {e}")

    logging.info("Hoàn tất việc cắt video Baseline!")

# Alias tương thích
cut_mobilenet_regions = cut_adult_regions

def main():
    parser = argparse.ArgumentParser(description="Gom cụm và cắt video từ kết quả Baseline")
    parser.add_argument("--json", type=str, required=True, help="Đường dẫn đến 12_final_decision.json")
    parser.add_argument("--out_dir", type=str, default=get_env("BASELINE_CLIPS_OUT_DIR", "baseline_bad_clips", resolve=True), help="Thư mục lưu")
    parser.add_argument("--video", type=str, default=None, help="(Tuỳ chọn) Ghi đè đường dẫn video gốc nếu file JSON trỏ sai")
    parser.add_argument("--pad", type=float, default=1.0, help="Mở rộng X giây ở hai đầu (Mặc định 1.0)")
    parser.add_argument("--gap", type=float, default=1.5, help="Khoảng cách tối đa giữa các frame để gom cụm (Mặc định 1.5)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    
    cut_adult_regions(
        json_path=args.json, 
        output_dir=args.out_dir, 
        override_video_path=args.video, 
        padding_sec=args.pad, 
        max_gap_sec=args.gap
    )

if __name__ == "__main__":
    main()
