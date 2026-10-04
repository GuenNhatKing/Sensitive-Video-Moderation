import sys
import argparse
import logging
from pathlib import Path

# Đưa thư mục src vào sys.path để import module dùng chung
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_config import get_env

# Import các module trong pipeline
from transnet_detector import SceneDetector          
from sampling import VideoSampler                    
from resnet_inference import ResNetClassifier        
from suspicious_region import SuspiciousRegionDetector 
from dense_sampling import DenseSampler              
from topk_temporal import FinalEvaluator             

def main():
    parser = argparse.ArgumentParser(description="Adult Video Detection Pipeline (In-memory)")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video")
    parser.add_argument("--out_dir", type=str, default=get_env("RESNET_OUT_DIR", "../ResnetLog", resolve=True), help="Thư mục lưu log")
    parser.add_argument("--transnet_weights", type=str, default=get_env("TRANSNET_WEIGHTS", "models/transnetv2-pytorch-weights.pth", resolve=True), help="Đường dẫn model transnet (.pth)")
    parser.add_argument("--transnet_dir", type=str, default=get_env("TRANSNET_DIR", "TransNetV2", resolve=True), help="Thư mục mã nguồn TransNetV2")
    parser.add_argument("--resnet_model", type=str, default=get_env("RESNET_MODEL", "models/resnet50_adult_normal_fcn_layer4_classifier.pth", resolve=True), help="Đường dẫn mô hình ResNet50 (.pth)")
    parser.add_argument("--early_stop", action="store_true", help="Dừng sớm ngay khi phát hiện vùng đầu tiên vi phạm Adult")
    parser.add_argument("--save_intermediate", action="store_true", help="Có lưu log đĩa ở từng bước không?")
    args = parser.parse_args()

    # Cấu hình logging toàn hệ thống (rút gọn thời gian, không in milis)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    
    video_path = Path(args.video).resolve()
    base_out = Path(args.out_dir) / video_path.stem
    
    logging.info(f"BẮT ĐẦU XỬ LÝ VIDEO: {video_path}")

    # 1. Khởi tạo tất cả các object (Load model 1 lần duy nhất)
    scene_det = SceneDetector(args.transnet_dir, args.transnet_weights)
    sampler = VideoSampler()
    resnet = ResNetClassifier(args.resnet_model)
    region_det = SuspiciousRegionDetector()
    dense_sampler = DenseSampler()
    evaluator = FinalEvaluator(sampling_interval_sec=dense_sampler.sampling_interval_sec)

    # 2. TransNet Cắt cảnh (Bước 02)
    shots_info, fps, frame_count = scene_det.detect_scenes(
        video_path, 
        output_json_path=base_out / "02_shots.json" if args.save_intermediate else None
    )

    # 3. Sparse Sampling (Bước 03)
    sparse_samples = sampler.sample_shots(
        video_path, shots_info, 
        output_dir=base_out / "03_sparse" if args.save_intermediate else None, 
        save_to_disk=args.save_intermediate
    )

    # 4. Sparse ResNet Inference (Bước 04)
    sparse_results = resnet.predict_samples(
        sparse_samples, 
        output_json_path=base_out / "04_resnet.json" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate
    )

    # 5. Phân tích vùng đáng ngờ (Bước 05)
    regions = region_det.detect_regions(
        sparse_results, 
        output_json_path=base_out / "05_regions.json" if args.save_intermediate else None,
        save_to_disk=args.save_intermediate
    )

    if not regions:
        logging.info("Video an toàn ngay từ bước 5. Kết thúc sớm.")
        evaluator.evaluate([], video_path, base_out / "08_final_decision.json", save_to_disk=True)
        return

    # Tách các vùng nghi vấn:
    # 1. direct_adult_regions: Đạt ngưỡng tin cậy cao -> Xác nhận Adult ngay, bỏ qua Dense Sampling
    # 2. dense_candidate_regions: Dưới ngưỡng tin cậy cao -> Cần Dense Sampling để kiểm tra tính nhất quán
    direct_adult_regions = [r for r in regions if r.get("is_direct_adult", False)]
    dense_candidate_regions = [r for r in regions if not r.get("is_direct_adult", False)]

    for r in direct_adult_regions:
        logging.info(
            f"REGION {r['region_id']:02d} (Shot {r['shot_id']:02d}) | "
            f"Peak Adult = {r['peak_probability']:.4f} vượt ngưỡng tin cậy cao => XÁC NHẬN ADULT (Bỏ qua Dense Sampling)"
        )

    # Nếu bật Early Stop và đã có vùng direct_adult: Kết thúc ngay không cần Dense bất kỳ vùng nào khác
    if args.early_stop and direct_adult_regions:
        logging.info("Phát hiện vùng vi phạm tin cậy cao (Direct Adult). [Early Stop kích hoạt: Dừng sớm pipeline]")
        final_result = evaluator.evaluate(
            [], 
            video_path, 
            output_json_path=base_out / "08_final_decision.json",
            save_to_disk=True,
            direct_adult_regions=direct_adult_regions,
            log_summary=True
        )
        logging.info(f"HOÀN TẤT PIPELINE. Kết luận: {final_result['final_decision'].upper()}")
        return

    # Nếu bật Early Stop và có các vùng candidate cần dense: Quét tuần tự từng vùng
    if args.early_stop and dense_candidate_regions:
        logging.info(f"Kích hoạt Early Stop: Quét tuần tự {len(dense_candidate_regions)} vùng nghi vấn...")
        accumulated_dense_results = []
        for cand in dense_candidate_regions:
            cand_id = cand["region_id"]
            dense_cand = dense_sampler.sample_regions(
                video_path, [cand], shots_info, sparse_results,
                output_dir=base_out / f"06_dense_r{cand_id}" if args.save_intermediate else None,
                save_to_disk=args.save_intermediate
            )
            
            flattened_dense = []
            for r in dense_cand:
                for f in r["frames"]:
                    sample = dict(f)
                    sample["region_id"] = cand_id
                    flattened_dense.append(sample)
                    
            cand_dense_results = resnet.predict_samples(
                flattened_dense,
                output_json_path=base_out / f"07_dense_resnet_r{cand_id}.json" if args.save_intermediate else None,
                save_to_disk=args.save_intermediate
            )
            accumulated_dense_results.extend(cand_dense_results)
            
            combined_cand = list(cand_dense_results)
            for f in cand["frames"]:
                f_copy = dict(f)
                f_copy["region_id"] = cand_id
                combined_cand.append(f_copy)
                
            f_sorted = sorted(combined_cand, key=lambda x: x["time_sec"])
            s_time = f_sorted[0]["time_sec"]
            e_time = f_sorted[-1]["time_sec"]
            logging.info(f"- Đã gộp Region {cand_id:02d}: {s_time:.2f}s -> {e_time:.2f}s ({len(combined_cand)} frames)")
            
            eval_res = evaluator.evaluate(
                combined_cand,
                video_path,
                output_json_path=None,
                save_to_disk=False,
                direct_adult_regions=direct_adult_regions,
                log_summary=False
            )
            
            if eval_res["final_decision"] == "adult":
                logging.info(f"Phát hiện Region {cand_id:02d} là ADULT. [Early Stop kích hoạt: Dừng sớm pipeline]")
                all_processed_frames = list(accumulated_dense_results)
                for c in dense_candidate_regions:
                    if c["region_id"] <= cand_id:
                        for f in c["frames"]:
                            f_copy = dict(f)
                            f_copy["region_id"] = c["region_id"]
                            all_processed_frames.append(f_copy)
                            
                final_result = evaluator.evaluate(
                    all_processed_frames,
                    video_path,
                    output_json_path=base_out / "08_final_decision.json",
                    save_to_disk=True,
                    direct_adult_regions=direct_adult_regions,
                    log_summary=True
                )
                logging.info(f"HOÀN TẤT PIPELINE. Kết luận: {final_result['final_decision'].upper()}")
                return

        # Nếu đã quét hết mà không vùng nào vi phạm
        all_processed_frames = list(accumulated_dense_results)
        for cand in dense_candidate_regions:
            cand_id = cand["region_id"]
            for f in cand["frames"]:
                f_copy = dict(f)
                f_copy["region_id"] = cand_id
                all_processed_frames.append(f_copy)
                
        final_result = evaluator.evaluate(
            all_processed_frames,
            video_path,
            output_json_path=base_out / "08_final_decision.json",
            save_to_disk=True,
            direct_adult_regions=direct_adult_regions,
            log_summary=True
        )
        logging.info(f"HOÀN TẤT PIPELINE. Kết luận: {final_result['final_decision'].upper()}")
        return

    dense_results = []
    if dense_candidate_regions:
        logging.info(f"Tiến hành Dense Sampling cho {len(dense_candidate_regions)} vùng còn lại...")
        # 6. Dense Sampling (Bước 06)
        dense_regions = dense_sampler.sample_regions(
            video_path, dense_candidate_regions, shots_info, sparse_results,
            output_dir=base_out / "06_dense" if args.save_intermediate else None,
            save_to_disk=args.save_intermediate
        )

        # 7. Dense ResNet Inference (Bước 07)
        flattened_dense = []
        for r in dense_regions:
            for f in r["frames"]:
                sample = dict(f)
                sample["region_id"] = r["region_id"]
                flattened_dense.append(sample)

        dense_results = resnet.predict_samples(
            flattened_dense,
            output_json_path=base_out / "07_dense_resnet.json" if args.save_intermediate else None,
            save_to_disk=args.save_intermediate
        )
    else:
        logging.info("Tất cả các vùng nghi vấn đều vượt ngưỡng tin cậy cao. Bỏ qua hoàn toàn Dense Sampling.")

    # Gộp kết quả thưa và dày của các vùng cần đánh giá Dense
    combined_results = list(dense_results)
    for r in dense_candidate_regions:
        region_id = r["region_id"]
        for frame in r["frames"]:
            frame_copy = dict(frame)
            frame_copy["region_id"] = region_id
            combined_results.append(frame_copy)

    # Hiển thị thông tin các vùng sau khi đã gộp mẫu thưa và dày
    region_frames_map = {}
    for res in combined_results:
        reg_id = res.get("region_id", 0)
        if reg_id not in region_frames_map:
            region_frames_map[reg_id] = []
        region_frames_map[reg_id].append(res)

    if region_frames_map:
        logging.info(f"Đã gộp mẫu thưa và dày cho {len(region_frames_map)} vùng:")
        for reg_id, f_list in sorted(region_frames_map.items()):
            f_sorted = sorted(f_list, key=lambda x: x["time_sec"])
            s_time = f_sorted[0]["time_sec"]
            e_time = f_sorted[-1]["time_sec"]
            logging.info(f"- Region {reg_id:02d}: {s_time:.2f}s -> {e_time:.2f}s ({len(f_list)} frames)")

    # 8. Đánh giá kết quả cuối cùng (Bước 08)
    final_result = evaluator.evaluate(
        combined_results, 
        video_path, 
        output_json_path=base_out / "08_final_decision.json",
        save_to_disk=True, # Luôn lưu file kết quả cuối cùng
        direct_adult_regions=direct_adult_regions,
        log_summary=True
    )

    logging.info(f"HOÀN TẤT PIPELINE. Kết luận: {final_result['final_decision'].upper()}")

if __name__ == "__main__":
    main()