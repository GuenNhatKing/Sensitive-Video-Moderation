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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_config import get_env, resolve_video_path

def cut_adult_regions(json_path, output_dir=None, override_video_path=None, padding_sec=0.0):
    """
    Đọc file JSON và cắt các đoạn video được đánh dấu là 'adult'.
    Tự động xác định đường dẫn video từ JSON mà không cần truyền --video.
    """
    logging.info(f"Đọc file JSON: {json_path}")
    
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

    adult_regions = data.get("adult_regions", [])
    if not adult_regions:
        logging.info("Tuyệt vời! Video này không có vùng 'adult' nào để cắt.")
        return

    logging.info(f"Tìm thấy {len(adult_regions)} vùng 'adult'. Bắt đầu cắt...")

    # Cấu hình thư mục đầu ra
    if not output_dir:
        output_dir = get_env("RESNET_CLIPS_OUT_DIR", "../resnet_bad_clips")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    for idx, region in enumerate(adult_regions):
        region_id = region.get("region_id", idx)
        start_sec = region.get("consistency_start_sec")
        end_sec = region.get("consistency_end_sec")

        if start_sec is None or end_sec is None:
            logging.warning(f"Region {region_id} bị thiếu thời gian start/end. Bỏ qua.")
            continue

        # Mở rộng thêm 1 chút thời gian (padding) nếu muốn xem bao quát hơn
        start_sec = max(0, start_sec - padding_sec)
        end_sec = end_sec + padding_sec

        # Tạo tên file đầu ra
        out_filename = f"{video_name}_region_{region_id:03d}_{start_sec:.1f}s_to_{end_sec:.1f}s.mp4"
        out_filepath = os.path.join(output_dir, out_filename)

        # Lệnh FFmpeg cắt chính xác (re-encode video để tránh lỗi frame đen ở đầu clip)
        cmd = [
            "ffmpeg", 
            "-y",                   # Ghi đè nếu file đã tồn tại
            "-ss", str(start_sec),  # Thời gian bắt đầu
            "-to", str(end_sec),    # Thời gian kết thúc
            "-i", str(video_path),  # File đầu vào
            "-c:v", "libx264",      # Encode video bằng H.264
            "-preset", "fast",      # Tốc độ xử lý nhanh
            "-c:a", "aac",          # Encode audio bằng AAC
            out_filepath
        ]

        logging.info(f"Đang cắt Region {region_id:02d} ({start_sec:.2f}s -> {end_sec:.2f}s)...")
        
        try:
            # Ẩn output rác của FFmpeg, chỉ hiện khi có lỗi
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            logging.info(f"-> Đã lưu: {out_filepath}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Lỗi khi cắt Region {region_id}: {e}")

    logging.info("Hoàn tất việc cắt video!")

def main():
    parser = argparse.ArgumentParser(description="Cắt các phân đoạn Adult từ video gốc (ResNet)")
    parser.add_argument("--json", type=str, required=True, help="Đường dẫn đến file 08_final_decision.json")
    parser.add_argument("--out_dir", type=str, default=get_env("RESNET_CLIPS_OUT_DIR", "../resnet_bad_clips", resolve=True), help="Thư mục lưu các clip được cắt")
    parser.add_argument("--video", type=str, default=None, help="(Tuỳ chọn) Ghi đè đường dẫn video gốc nếu file JSON trỏ sai")
    parser.add_argument("--pad", type=float, default=0.0, help="Mở rộng thêm X giây ở đầu và cuối mỗi clip")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    
    cut_adult_regions(
        json_path=args.json,
        output_dir=args.out_dir,
        override_video_path=args.video,
        padding_sec=args.pad
    )

if __name__ == "__main__":
    main()