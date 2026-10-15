import sys
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

from env_config import get_env

# Import các module trong pipeline
from extract_frames import FrameExtractor               
from ssim_filter import SSIMFilter                     
from baseline.mobilenet_inference import MobileNetClassifier    
from baseline_evaluator import MobileNetEvaluator     

def main():
    parser = argparse.ArgumentParser(description="Privacy-preserving On-device Video Detection (MobileNet)")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video")
    parser.add_argument("--out_dir", type=str, default=get_env("BASELINE_OUT_DIR", "BaselineLog", resolve=True), help="Thư mục lưu log/kết quả")
    parser.add_argument("--model", type=str, default=get_env("MOBILENET_MODEL", "models/mobilenetv3_small_adult_normal.keras", resolve=True), help="Đường dẫn mô hình MobileNet (.keras)")
    parser.add_argument("--ssim", type=float, default=0.95, help="Ngưỡng SSIM để loại bỏ frame")
    parser.add_argument("--ssim_size", type=int, default=224, help="Kích thước thu nhỏ ảnh trước khi tính SSIM (mặc định: 224)")
    parser.add_argument("--adult_threshold", type=float, default=0.85, help="Ngưỡng xác suất phân loại Adult")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size xử lý frame")
    parser.add_argument("--early_stop", action="store_true", help="Dừng sớm ngay khi phát hiện frame đầu tiên vi phạm Adult")
    parser.add_argument("--save_intermediate", action="store_true", help="Lưu đĩa các file ảnh/json để debug")
    args = parser.parse_args()

    # Cấu hình logging toàn hệ thống (rút gọn thời gian, không in milis)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    
    video_path = Path(args.video).resolve()
    base_out = Path(args.out_dir) / video_path.stem
    
    logging.info(f"BẮT ĐẦU XỬ LÝ VIDEO: {video_path}")

    # 1. Khởi tạo các object
    extractor = FrameExtractor()
    ssim_filter = SSIMFilter(ssim_threshold=args.ssim, target_size=(args.ssim_size, args.ssim_size))
    classifier = MobileNetClassifier(model_path=args.model, batch_size=args.batch_size, adult_threshold=args.adult_threshold)
    evaluator = MobileNetEvaluator()

    # 2. Xây dựng đường ống dữ liệu (Data Piping)
    # Bước 9: Tạo luồng đọc từng frame từ video
    raw_stream = extractor.extract_stream(
        video_path=video_path, 
        output_dir=base_out / "09_raw_frames" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate
    )

    # Bước 10: Nối luồng raw vào bộ lọc SSIM
    filtered_stream = ssim_filter.filter_stream(
        frame_generator=raw_stream,
        output_dir=base_out / "10_ssim_frames" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate
    )

    # Bước 11: Nối luồng đã lọc vào MobileNet để suy luận theo batch
    inference_results = classifier.predict_stream(
        ssim_stream=filtered_stream,
        output_json_path=base_out / "11_inference.json" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate,
        early_stop=args.early_stop
    )

    # Bước 12: Đánh giá kết quả cuối cùng (Dựa vào mảng kết quả trả về từ B11)
    final_result = evaluator.evaluate(
        inference_results=inference_results,
        video_path=video_path,
        output_json_path=base_out / "12_final_decision.json",
        save_to_disk=True # Luôn lưu file chốt cuối cùng
    )

    logging.info(f"HOÀN TẤT PIPELINE ON-DEVICE. KẾT LUẬN: {final_result['final_decision'].upper()}")

if __name__ == "__main__":
    main()