import json
import logging
import os
from pathlib import Path


class DualBranchEvaluator:
    """
    THAY ĐỔI 4: Bộ Đánh giá Hợp nhất Hai Nhánh
    (Thay thế quy tắc Dừng sớm 0.97 và Top-K cũ)
    - Tách biệt hoàn toàn luồng phát hiện "Phân cảnh nhạy cảm kéo dài" (NHÁNH 1)
      và luồng "Chèn 1-2 frames nhấp nháy lách luật" (NHÁNH 2).
    - Đạt Recall = 100% đồng thời ép False Positive (FP) -> 0.
    """

    def __init__(
        self,
        sustained_mean_threshold=0.85,
        sustained_valid_frame_threshold=0.80,
        sustained_valid_ratio_threshold=0.60,
        flash_z_threshold=3.0,
        flash_neighbors_max_threshold=0.20,
        flash_mobile_threshold=0.90,
        flash_resnet_threshold=0.95,
        flash_crop_threshold=0.95
    ):
        self.sustained_mean_threshold = sustained_mean_threshold
        self.sustained_valid_frame_threshold = sustained_valid_frame_threshold
        self.sustained_valid_ratio_threshold = sustained_valid_ratio_threshold

        self.flash_z_threshold = flash_z_threshold
        self.flash_neighbors_max_threshold = flash_neighbors_max_threshold
        self.flash_mobile_threshold = flash_mobile_threshold
        self.flash_resnet_threshold = flash_resnet_threshold
        self.flash_crop_threshold = flash_crop_threshold

        logging.info("Khởi tạo DualBranchEvaluator (Nhánh 1: Sustained NSFW | Nhánh 2: Flash Insert)")

    def evaluate_anchor(self, anchor, d_star_frames, resnet_classifier):
        """
        Đánh giá một tâm điểm nghi vấn t_0 dựa trên tập 5 frames D* và 2 nhánh kiểm định.
        """
        t_0 = anchor["time_sec"]
        p_mobile_t0 = anchor["p_mobile"]
        z_flash_t0 = anchor["z_flash"]

        # 1. Dự đoán xác suất ResNet-50 FCN: Gom toàn bộ 5 frames trong D* (Input 360x640)
        # và 1 frame CenterCrop của anchor f_{t_0} vào đúng 1 Batch 6 frames duy nhất (1 forward pass)
        frame_data_list = [d["frame_data"] for d in d_star_frames]
        anchor_frame_data = anchor.get("frame_data")

        if hasattr(resnet_classifier, "predict_anchor_batch"):
            p_resnet_list, p_crop_t0 = resnet_classifier.predict_anchor_batch(
                frame_data_list, anchor_frame_data
            )
        else:
            p_resnet_list = resnet_classifier.predict_frame_batch(frame_data_list, is_center_crop=False)
            p_crop_t0 = 0.0

        d_eval_list = []
        anchor_d_item = None
        p_resnet_t0 = 0.0

        for d_item, p_res in zip(d_star_frames, p_resnet_list):
            eval_item = {
                "frame_index": d_item["frame_index"],
                "time_sec": d_item["time_sec"],
                "is_anchor": d_item["is_anchor"],
                "p_resnet": round(p_res, 6)
            }
            d_eval_list.append(eval_item)
            if d_item["is_anchor"]:
                anchor_d_item = d_item
                p_resnet_t0 = p_res

        if anchor_d_item is None and d_eval_list:
            # Fallback nếu không gắn cờ is_anchor
            anchor_d_item = d_star_frames[0]
            p_resnet_t0 = p_resnet_list[0]

        num_d = len(d_eval_list)
        if num_d == 0:
            return {
                "anchor_time_sec": t_0,
                "decision": "NORMAL",
                "branch_1": False,
                "branch_2": False,
                "crop_evaluated": False
            }

        # ============================================================
        # NHÁNH 1: Kiểm định Vi phạm Bền vững (Sustained NSFW Branch)
        # ============================================================
        p_mean = float(sum(p_resnet_list) / num_d)
        count_valid = sum(1 for p in p_resnet_list if p >= self.sustained_valid_frame_threshold)
        valid_ratio = float(count_valid / num_d)

        cond_branch_1 = bool(
            (p_mean >= self.sustained_mean_threshold) and
            (valid_ratio >= self.sustained_valid_ratio_threshold)
        )

        # ============================================================
        # NHÁNH 2: Kiểm định Chèn Nhấp nháy Lách luật (Flash-Insert Branch)
        # ============================================================
        # P_neighbors_max(t_0) = max_{f in D* \ {f_{t_0}}} p_resnet(f)
        neighbor_probs = [
            it["p_resnet"] for it in d_eval_list if not it["is_anchor"]
        ]
        p_neighbors_max = float(max(neighbor_probs)) if neighbor_probs else 0.0

        # Điều kiện (1) Đột biến cô lập : ( Z_flash(t_0) > 3.0 ) AND ( P_neighbors_max(t_0) < 0.20 )
        cond_2_1 = bool(
            (z_flash_t0 > self.flash_z_threshold) and
            (p_neighbors_max < self.flash_neighbors_max_threshold)
        )

        # Điều kiện (2) Đồng thuận chéo : ( p_mobile(t_0) >= 0.90 ) AND ( p_resnet(f_{t_0}) >= 0.95 )
        cond_2_2 = bool(
            (p_mobile_t0 >= self.flash_mobile_threshold) and
            (p_resnet_t0 >= self.flash_resnet_threshold)
        )

        # Điều kiện (3) Bất biến phóng đại: p_crop(t_0) = ResNet50_FCN( CenterCrop_360x640(f_{t_0}) ) >= 0.95
        # p_crop_t0 đã được tính sẵn trong cùng 1 batch với D*, không cần forward pass đơn lẻ
        cond_2_3 = False
        crop_evaluated = False
        if cond_2_1 and cond_2_2:
            crop_evaluated = True
            if p_crop_t0 == 0.0 and anchor_frame_data is not None:
                p_crop_t0 = resnet_classifier.predict_frame_data(
                    anchor_frame_data,
                    is_center_crop=True
                )
            cond_2_3 = bool(p_crop_t0 >= self.flash_crop_threshold)

        cond_branch_2 = bool(cond_2_1 and cond_2_2 and cond_2_3)

        # Hàm Phán quyết Cuối cùng cho Vùng nghi vấn t_0:
        # Decision(t_0) = ADULT nếu ( Branch_1 == True ) OR ( Branch_2 == True )
        is_adult = bool(cond_branch_1 or cond_branch_2)
        decision = "ADULT" if is_adult else "NORMAL"

        violation_type = None
        if cond_branch_1 and cond_branch_2:
            violation_type = "BOTH_SUSTAINED_AND_FLASH"
        elif cond_branch_1:
            violation_type = "SUSTAINED_NSFW"
        elif cond_branch_2:
            violation_type = "FLASH_INSERT"

        return {
            "anchor_frame_index": anchor["frame_index"],
            "anchor_time_sec": round(t_0, 4),
            "p_mobile": round(p_mobile_t0, 6),
            "z_flash": round(z_flash_t0, 4),
            "p_resnet_anchor": round(p_resnet_t0, 6),
            "decision": decision,
            "violation_type": violation_type,
            "branch_1": {
                "triggered": cond_branch_1,
                "p_mean": round(p_mean, 6),
                "count_valid": count_valid,
                "total_frames": num_d,
                "valid_ratio": round(valid_ratio, 4)
            },
            "branch_2": {
                "triggered": cond_branch_2,
                "p_neighbors_max": round(p_neighbors_max, 6),
                "p_crop": round(p_crop_t0, 6),
                "cond_isolation": cond_2_1,
                "cond_cross_consensus": cond_2_2,
                "cond_zoom_invariance": cond_2_3
            },
            "d_star": d_eval_list,
            "crop_evaluated": crop_evaluated
        }

    def group_adult_regions(self, adult_anchors, max_gap_sec=2.0):
        """
        Gom nhóm các mốc anchor vi phạm gần nhau thành các vùng liên tục để cắt clip.
        Định dạng đầu ra chuẩn tương thích 100% với cut_adult_regions.py.
        """
        if not adult_anchors:
            return []

        sorted_anchors = sorted(adult_anchors, key=lambda a: a["anchor_time_sec"])
        regions = []
        current_region = [sorted_anchors[0]]

        for i in range(1, len(sorted_anchors)):
            prev = current_region[-1]
            curr = sorted_anchors[i]
            if curr["anchor_time_sec"] - prev["anchor_time_sec"] <= max_gap_sec:
                current_region.append(curr)
            else:
                regions.append(current_region)
                current_region = [curr]
        if current_region:
            regions.append(current_region)

        formatted_regions = []
        for reg_id, group in enumerate(regions, start=1):
            start_sec = max(0.0, min(a["anchor_time_sec"] for a in group) - 0.5)
            end_sec = max(a["anchor_time_sec"] for a in group) + 0.5
            peak_prob = max(a["p_resnet_anchor"] for a in group)
            v_types = list(set(a["violation_type"] for a in group if a["violation_type"]))

            formatted_regions.append({
                "region_id": reg_id,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "consistency_start_sec": round(start_sec, 3),
                "consistency_end_sec": round(end_sec, 3),
                "peak_probability": round(peak_prob, 4),
                "violation_type": ", ".join(v_types),
                "total_anchors": len(group),
                "anchors": [
                    {
                        "time_sec": a["anchor_time_sec"],
                        "frame_index": a["anchor_frame_index"],
                        "p_resnet": a["p_resnet_anchor"],
                        "type": a["violation_type"]
                    }
                    for a in group
                ]
            })

        return formatted_regions

    def evaluate_video(
        self,
        video_path,
        anchor_evaluations,
        video_info,
        output_json_path=None,
        save_to_disk=True
    ):
        """
        Tổng hợp kết quả toàn video và lưu JSON chi tiết.
        """
        video_path = Path(video_path).resolve()
        adult_anchors = [a for a in anchor_evaluations if a["decision"] == "ADULT"]
        final_decision = "ADULT" if adult_anchors else "NORMAL"

        adult_regions = self.group_adult_regions(adult_anchors)

        sustained_count = sum(1 for a in adult_anchors if a["branch_1"]["triggered"])
        flash_count = sum(1 for a in adult_anchors if a["branch_2"]["triggered"])

        summary = {
            "video": str(video_path),
            "final_decision": final_decision,
            "total_anchors_evaluated": len(anchor_evaluations),
            "adult_anchors_count": len(adult_anchors),
            "sustained_violations_count": sustained_count,
            "flash_violations_count": flash_count,
            "total_adult_regions": len(adult_regions),
            "video_metadata": {
                "total_frames": video_info.get("total_frames"),
                "fps": video_info.get("fps"),
                "duration_sec": video_info.get("duration_sec"),
                "stage1_selected_frames": video_info.get("selected_frames"),
                "stage1_flash_candidates": video_info.get("flash_candidates"),
                "total_shots": len(video_info.get("shots", []))
            },
            "adult_regions": adult_regions,
            "anchor_details": anchor_evaluations
        }

        if save_to_disk and output_json_path:
            out_p = Path(output_json_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logging.info(f"Đã lưu kết quả phán quyết cuối cùng: {out_p}")

        return summary

