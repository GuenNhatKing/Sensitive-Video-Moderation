import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Đưa thư mục src vào sys.path để import module dùng chung
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_config import get_env

# Khởi tạo PyTorch & torchvision trước để tránh xung đột dynamic linker với TensorFlow runtime
import torch
import torchvision
from proposed.resnet_inference import ResNetClassifier
from proposed.diversity_spatial_sampler import DiversitySpatialSampler
from proposed.compressed_color_sampler import CompressedColorSampler
from proposed.coarse_mobilenet_filter import CoarseMobileNetFilter
from proposed.dual_branch_evaluator import DualBranchEvaluator

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="PBL6 Proposed Sensitive Video Moderation Pipeline (Cascade 4 Stages)"
    )
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video cần phân tích")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=get_env("PROPOSED_OUT_DIR", get_env("RESNET_OUT_DIR", "../ProposedLog"), resolve=True),
        help="Thư mục lưu log và kết quả phán quyết"
    )
    parser.add_argument(
        "--mobilenet_model",
        type=str,
        default=get_env("MOBILENET_MODEL", "models/mobilenetv3_small_adult_normal.keras", resolve=True),
        help="Đường dẫn mô hình MobileNetV3-Small (.keras)"
    )
    parser.add_argument(
        "--resnet_model",
        type=str,
        default=get_env("PROPOSED_MODEL", get_env("RESNET_MODEL", "models/resnet50_adult_normal_fcn_layer4_classifier.pth"), resolve=True),
        help="Đường dẫn mô hình ResNet50 FCN (.pth)"
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=0.5,
        help="Khoảng cách thời gian tối đa giữa các frame lấy mẫu cơ sở (mặc định: 0.5s)"
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=3.0,
        help="Ngưỡng Z-score phát hiện đột biến flash nhấp nháy (mặc định: 3.0)"
    )
    parser.add_argument(
        "--rejection_threshold",
        type=float,
        default=0.35,
        help="Ngưỡng loại bỏ sớm của MobileNetV3-Small ở Bước 2 (mặc định: 0.35)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size chạy quét thô MobileNetV3 (mặc định: 32)"
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Dừng sớm ngay khi phát hiện mốc vi phạm Adult đầu tiên"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Lưu kết quả JSON và frames trung gian ở từng giai đoạn"
    )
    args = parser.parse_args()

    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    video_path = Path(args.video).resolve()
    base_out = Path(args.out_dir) / video_path.stem

    logging.info(f"BẮT ĐẦU PIPELINE ĐỀ XUẤT (PROPOSED PIPELINE): {video_path.name}")
    start_total_time = time.time()

    # ============================================================
    # THAY ĐỔI 1: Khai thác Miền nén & Biến thiên Màu sắc
    # ============================================================
    t0_stage1 = time.time()
    stage1_sampler = CompressedColorSampler(
        delta_t=args.delta_t,
        z_threshold=args.z_threshold,
        window_size=15
    )
    samples, shots, video_info = stage1_sampler.sample_video(
        video_path=video_path,
        output_dir=base_out / "01_stage1_frames" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate
    )
    logging.info(f">> Stage 1 hoàn tất trong {time.time() - t0_stage1:.2f}s.")

    if not samples:
        logging.warning("Không trích xuất được frame nào từ video. Kết luận: NORMAL.")
        evaluator = DualBranchEvaluator()
        evaluator.evaluate_video(
            video_path=video_path,
            anchor_evaluations=[],
            video_info=video_info,
            output_json_path=base_out / "08_final_decision.json",
            save_to_disk=True
        )
        return

    # ============================================================
    # THAY ĐỔI 2: Quét thô Độ nhạy cao bằng MobileNetV3-Small
    # ============================================================
    t0_stage2 = time.time()
    stage2_filter = CoarseMobileNetFilter(
        model_path=args.mobilenet_model,
        batch_size=args.batch_size,
        rejection_threshold=args.rejection_threshold
    )
    all_stage2_results, anchor_set = stage2_filter.filter_candidates(
        samples=samples,
        output_json_path=base_out / "02_stage2_coarse.json" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate
    )
    logging.info(f">> Stage 2 hoàn tất trong {time.time() - t0_stage2:.2f}s.")

    evaluator = DualBranchEvaluator()

    # Early Rejection: Nếu không có mốc nào vượt ngưỡng 0.35 -> Kết luận NORMAL ngay lập tức
    if not anchor_set:
        logging.info("TẤT CẢ CÁC FRAMES ĐÃ BỊ LOẠI BỎ SỚM TẠI STAGE 2 (p_mobile < 0.35) -> VIDEO AN TOÀN (NORMAL).")
        evaluator.evaluate_video(
            video_path=video_path,
            anchor_evaluations=[],
            video_info=video_info,
            output_json_path=base_out / "08_final_decision.json",
            save_to_disk=True
        )
        return

    # ============================================================
    # THAY ĐỔI 3 & 4: Phóng đại Không gian (360x640), Chọn 4 Frame Đa dạng Cực đại
    # & Bộ Đánh giá Hợp nhất Hai Nhánh
    # ============================================================
    t0_stage3_4 = time.time()
    logging.info("Khởi tạo ResNet-50 FCN cho Stage 3 và Stage 4...")
    resnet = ResNetClassifier(model_path=args.resnet_model)
    diversity_sampler = DiversitySpatialSampler(time_window_radius=1.0, sampling_fps=10.0, target_num_frames=5)

    cap = cv2.VideoCapture(str(video_path))
    fps = video_info["fps"]
    duration_sec = video_info["duration_sec"]

    anchor_evaluations = []
    has_adult = False

    logging.info(f"Bắt đầu đánh giá hai nhánh cho {len(anchor_set)} tâm điểm nghi vấn (Anchors)...")

    for idx, anchor in enumerate(anchor_set, start=1):
        t_0 = anchor["time_sec"]

        # Stage 3: Chọn tập D* (5 frames) có độ đa dạng cực đại
        d_star = diversity_sampler.extract_diverse_frames(
            cap=cap,
            fps=fps,
            anchor=anchor,
            video_duration_sec=duration_sec
        )

        # Stage 4: Đánh giá hợp nhất hai nhánh
        anchor_eval = evaluator.evaluate_anchor(
            anchor=anchor,
            d_star_frames=d_star,
            resnet_classifier=resnet
        )
        anchor_evaluations.append(anchor_eval)

        status_str = f"[{idx:02d}/{len(anchor_set):02d}] Anchor {anchor['frame_index']} ({t_0:.2f}s) | "
        if anchor_eval["decision"] == "ADULT":
            has_adult = True
            v_type = anchor_eval["violation_type"]
            logging.info(
                f"{status_str}==> VI PHẠM ADULT ({v_type}) | "
                f"p_mobile: {anchor['p_mobile']:.3f}, p_resnet: {anchor_eval['p_resnet_anchor']:.3f}, "
                f"Z_flash: {anchor['z_flash']:.2f}"
            )
            if args.early_stop:
                logging.info("Kích hoạt --early_stop: Dừng sớm ngay khi phát hiện vi phạm Adult.")
                break
        else:
            logging.debug(f"{status_str}NORMAL | p_mobile: {anchor['p_mobile']:.3f}, p_resnet: {anchor_eval['p_resnet_anchor']:.3f}")

    cap.release()
    logging.info(f">> Stage 3 & 4 hoàn tất trong {time.time() - t0_stage3_4:.2f}s.")

    # ============================================================
    # Tổng kết & Lưu báo cáo
    # ============================================================
    final_result = evaluator.evaluate_video(
        video_path=video_path,
        anchor_evaluations=anchor_evaluations,
        video_info=video_info,
        output_json_path=base_out / "08_final_decision.json",
        save_to_disk=True
    )

    # Lưu thêm 04_final_decision.json để chuẩn hóa theo tên stage
    if args.save_intermediate:
        with open(base_out / "04_final_decision.json", "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - start_total_time
    logging.info(
        f"KẾT LUẬN TOÀN BỘ VIDEO: {final_result['final_decision']} "
        f"(Tổng thời gian xử lý: {total_elapsed:.2f}s | "
        f"Số vùng vi phạm: {final_result['total_adult_regions']})"
    )


if __name__ == "__main__":
    main()
