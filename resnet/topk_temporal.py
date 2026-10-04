import json
import logging
import os
from pathlib import Path
from collections import defaultdict

class FinalEvaluator:
    def __init__(self, 
                 top_k_ratio=0.15, min_top_k=3, 
                 adult_threshold=0.85, 
                 flash_threshold=0.97,
                 consistent_ratio=0.10, min_consistent_frames=3, 
                 sampling_interval_sec=0.10):
        """
        top_k_ratio: Lấy bao nhiêu % tổng số frame của vùng để làm Top-K (VD: 0.15 = 15%).
        min_top_k: Ít nhất phải lấy bao nhiêu frame (để chống lại vùng quá ngắn).
        consistent_ratio: Yêu cầu chuỗi liên tiếp phải dài ít nhất bằng bao nhiêu % tổng số frame.
        min_consistent_frames: Yêu cầu số frame liên tiếp tối thiểu.
        flash_threshold: Ngưỡng nhận diện tin cậy cao để xác nhận Adult trực tiếp mà không cần chuỗi liên tục.
        sampling_interval_sec: Chu kỳ lấy mẫu dày (khoảng cách thời gian giữa 2 frame dense, mặc định 0.10s [10 FPS]).
        """
        self.top_k_ratio = top_k_ratio
        self.min_top_k = min_top_k
        
        self.adult_threshold = adult_threshold
        self.flash_threshold = flash_threshold
        
        self.consistent_ratio = consistent_ratio
        self.min_consistent_frames = min_consistent_frames
        
        self.sampling_interval_sec = sampling_interval_sec
        
        logging.info("Khởi tạo Final Evaluator (Dynamic Logic)")
        logging.info(f"- Top-K: {self.top_k_ratio*100}% frames (Min: {self.min_top_k})")
        logging.info(f"- Consistent: {self.consistent_ratio*100}% frames (Min: {self.min_consistent_frames})")
        logging.info(f"- Ngưỡng Adult: {self.adult_threshold} | Ngưỡng Flash/High: {self.flash_threshold}")
        logging.info(f"- Khoảng cách lấy mẫu dày: {self.sampling_interval_sec}s/frame")

    def _find_temporal_consistency(self, region_results, required_consistent_frames, max_gap_sec):
        """
        Tìm chuỗi liên tiếp vượt ngưỡng.
        RÀNG BUỘC: Không gộp chuỗi nếu thuộc 2 shot khác nhau.
        max_gap_sec: Được tính động dựa trên required_consistent_frames * sampling_interval_sec.
        """
        sorted_results = sorted(region_results, key=lambda x: x["time_sec"])
        best_run = []
        current_run = []
        previous_time = None
        previous_shot = None

        for result in sorted_results:
            prob = result["adult_probability"]
            time_sec = result["time_sec"]
            shot_id = result.get("shot_id")

            if prob < self.adult_threshold:
                if len(current_run) > len(best_run): best_run = current_run
                current_run = []
                previous_time = None
                previous_shot = None
                continue

            if not current_run:
                current_run = [result]
                previous_time = time_sec
                previous_shot = shot_id
                continue

            gap = time_sec - previous_time
            # Ràng buộc: Khoảng cách <= max_gap_sec VÀ cùng thuộc một shot_id
            same_shot = (shot_id == previous_shot) if (shot_id is not None and previous_shot is not None) else True
            if gap <= max_gap_sec and same_shot:
                current_run.append(result)
            else:
                if len(current_run) > len(best_run): best_run = current_run
                current_run = [result]
            previous_time = time_sec
            previous_shot = shot_id

        if len(current_run) > len(best_run):
            best_run = current_run

        is_consistent = len(best_run) >= required_consistent_frames
        return is_consistent, best_run

    def evaluate(self, dense_results, video_path, output_json_path=None, save_to_disk=True, direct_adult_regions=None, log_summary=True):
        video_abs_path = str(Path(video_path).resolve())
        direct_adult_regions = direct_adult_regions or []
        
        region_evaluations = []
        adult_regions = []

        # 0. Nạp các vùng đã xác nhận Adult trực tiếp từ bước Sparse (vượt ngưỡng tin cậy cao)
        for r in direct_adult_regions:
            start_sec = r["start_time_sec"]
            end_sec = r["end_time_sec"]
            if end_sec <= start_sec:
                end_sec = round(start_sec + 0.3, 4)
                
            total_frames = r.get("frame_count", len(r.get("frames", [])))
            adult_frames = sum(1 for f in r.get("frames", []) if f.get("adult_probability", 0.0) >= self.adult_threshold)
            if adult_frames == 0 and r.get("peak_probability", 0.0) > self.flash_threshold:
                adult_frames = 1
            normal_frames = max(0, total_frames - adult_frames)
            adult_ratio_pct = round((adult_frames / total_frames) * 100, 2) if total_frames > 0 else 0.0
                
            direct_data = {
                "region_id": r["region_id"],
                "shot_id": r["shot_id"],
                "total_frames": total_frames,
                "adult_frames": adult_frames,
                "normal_frames": normal_frames,
                "adult_ratio_pct": adult_ratio_pct,
                "peak_probability": round(r["peak_probability"], 6),
                "applied_top_k": total_frames,
                "top_k_mean": round(r["peak_probability"], 6),
                "applied_min_consistent": 1,
                "temporal_consistent": True,
                "consistent_frame_count": total_frames,
                "consistency_start_sec": start_sec,
                "consistency_end_sec": end_sec,
                "is_adult": True,
                "decision": "adult",
                "reason": f"High confidence peak ({r['peak_probability']:.4f}) - skipped dense sampling"
            }
            region_evaluations.append(direct_data)
            adult_regions.append(direct_data)

        if not dense_results and not direct_adult_regions:
            logging.info("Không có kết quả Dense Inference và không có Direct Adult. Video: NORMAL.")
            result_dict = {
                "video": video_abs_path,
                "final_decision": "normal",
                "total_regions": 0,
                "adult_region_count": 0,
                "adult_regions": [],
                "regions": []
            }
            if save_to_disk and output_json_path:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
            return result_dict

        # 1. Bỏ trùng lặp frame cho phần kết quả Dense
        if dense_results:
            unique_frames = {}
            for res in dense_results:
                idx = res["frame_index"]
                if idx not in unique_frames or res["adult_probability"] > unique_frames[idx]["adult_probability"]:
                    unique_frames[idx] = res

            # 2. Nhóm theo region
            regions = defaultdict(list)
            for res in unique_frames.values():
                regions[res.get("region_id", 0)].append(res)

            # 3. Phân tích từng Region đã qua Dense
            for region_id, frames in sorted(regions.items()):
                total_frames = len(frames)
                shot_id = frames[0].get("shot_id") if frames else None
                
                # Thống kê số frame adult và normal trong region
                adult_frame_count = sum(1 for f in frames if f.get("adult_probability", 0.0) >= self.adult_threshold)
                normal_frame_count = total_frames - adult_frame_count
                adult_ratio_pct = round((adult_frame_count / total_frames) * 100, 2) if total_frames > 0 else 0.0

                # Dynamic Top-K
                k = max(self.min_top_k, int(total_frames * self.top_k_ratio))
                k = min(k, total_frames)
                
                # Dynamic Consistent Frames
                required_consistent = max(self.min_consistent_frames, int(total_frames * self.consistent_ratio))
                required_consistent = min(required_consistent, total_frames)

                # max_gap_sec tính động dựa trên: consistent_frames * sampling_interval_sec (VD: 0.1s/frame [10 FPS])
                max_gap_sec = round(required_consistent * self.sampling_interval_sec, 4)

                # Lọc Top-K
                sorted_frames = sorted(frames, key=lambda x: x["adult_probability"], reverse=True)
                top_k_mean = sum(r["adult_probability"] for r in sorted_frames[:k]) / k if k > 0 else 0
                peak_prob = sorted_frames[0]["adult_probability"] if sorted_frames else 0.0
                
                # Sắp xếp các frames theo thời gian thực tế của cả Region sau khi đã gộp thưa và dày
                sorted_by_time = sorted(frames, key=lambda x: x["time_sec"])
                region_start_sec = sorted_by_time[0]["time_sec"]
                region_end_sec = sorted_by_time[-1]["time_sec"]

                # Tìm chuỗi liên tục (đảm bảo cùng shot và max_gap_sec động)
                is_consistent, consistent_frames = self._find_temporal_consistency(
                    frames, required_consistent, max_gap_sec=max_gap_sec
                )
                
                # Phán quyết cho Region:
                # Nếu có frame vượt ngưỡng tin cậy cao (flash_threshold) -> Xác nhận Adult ngay
                # Ngược lại -> Yêu cầu điểm Top-K trung bình và chuỗi thời gian liên tiếp cùng đạt ngưỡng
                if peak_prob > self.flash_threshold:
                    region_is_adult = True
                    decision_reason = f"High confidence peak frame ({peak_prob:.4f})"
                else:
                    region_is_adult = (top_k_mean >= self.adult_threshold) and is_consistent
                    decision_reason = "Temporal consistency satisfied" if region_is_adult else "Inconsistent / below threshold"
                
                start_sec = consistent_frames[0]["time_sec"] if consistent_frames else sorted_frames[0]["time_sec"]
                end_sec = consistent_frames[-1]["time_sec"] if consistent_frames else sorted_frames[0]["time_sec"]
                if end_sec <= start_sec:
                    end_sec = round(start_sec + 0.3, 4)

                region_data = {
                    "region_id": region_id,
                    "shot_id": shot_id,
                    "start_time_sec": region_start_sec,
                    "end_time_sec": region_end_sec,
                    "total_frames": total_frames,
                    "adult_frames": adult_frame_count,
                    "normal_frames": normal_frame_count,
                    "adult_ratio_pct": adult_ratio_pct,
                    "peak_probability": round(peak_prob, 6),
                    "applied_top_k": k,
                    "top_k_mean": round(top_k_mean, 6),
                    "applied_min_consistent": required_consistent,
                    "applied_max_gap_sec": max_gap_sec,
                    "temporal_consistent": is_consistent,
                    "consistent_frame_count": len(consistent_frames),
                    "consistency_start_sec": start_sec,
                    "consistency_end_sec": end_sec,
                    "is_adult": bool(region_is_adult),
                    "decision": "adult" if region_is_adult else "normal",
                    "reason": decision_reason
                }
                region_evaluations.append(region_data)
                
                if region_is_adult:
                    adult_regions.append(region_data)

        # Sắp xếp lại danh sách các vùng theo region_id
        region_evaluations.sort(key=lambda x: x["region_id"])
        adult_regions.sort(key=lambda x: x["region_id"])

        # 4. Phán quyết toàn video
        video_is_adult = len(adult_regions) > 0
        final_decision = "adult" if video_is_adult else "normal"

        # In bảng tổng hợp chi tiết ra console (INFO level) nếu được bật
        if log_summary:
            logging.info("KẾT QUẢ ĐÁNH GIÁ TỪNG REGION")
            for r in region_evaluations:
                status_tag = "ADULT" if r["is_adult"] else "NORMAL"
                time_str = f"{r.get('start_time_sec', r.get('consistency_start_sec', 0.0)):.2f}s -> {r.get('end_time_sec', r.get('consistency_end_sec', 0.0)):.2f}s"
                logging.info(
                    f"- Region {r['region_id']:02d} (Shot {r.get('shot_id', '?')} | {time_str}): "
                    f"{r['adult_frames']}/{r['total_frames']} frames Adult ({r['adult_ratio_pct']}%) | "
                    f"Peak: {r.get('peak_probability', 0.0):.4f} | "
                    f"Đánh giá: {status_tag} [{r['reason']}]"
                )
            logging.info(
                f"TỔNG KẾT TOÀN VIDEO: {len(adult_regions)}/{len(region_evaluations)} vùng vi phạm "
                f"-> KẾT LUẬN: {final_decision.upper()}"
            )

        result_dict = {
            "video": video_abs_path,
            "final_decision": final_decision,
            "total_regions": len(region_evaluations),
            "adult_region_count": len(adult_regions),
            "adult_regions": adult_regions,
            "regions": region_evaluations
        }

        if save_to_disk and output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logging.info(f"Đã lưu kết quả Video-Level vào: {output_json_path}")
                
        return result_dict