import logging
import os
import json

def run_dense_inference(resnet_classifier, dense_regions, output_json_path=None, save_to_disk=True):
    """
    Sử dụng lại class ResNetClassifier để đánh giá các khung hình trích xuất dày đặc.
    - resnet_classifier: Object của class ResNetClassifier (đã load model từ bước 04).
    - dense_regions: Output từ hàm sample_regions của class DenseSampler.
    """
    if not dense_regions:
        logging.info("Không có dữ liệu dense regions để chạy inference.")
        return []

    # 1. Làm phẳng (Flatten) danh sách frame từ các regions
    flattened_samples = []
    for region in dense_regions:
        for frame in region["frames"]:
            # Copy frame info để không sửa trực tiếp vào biến gốc
            sample = dict(frame)
            sample["region_id"] = region["region_id"]
            flattened_samples.append(sample)

    logging.info(f"Bắt đầu chạy Dense Inference cho {len(flattened_samples)} frames...")

    # 2. Chạy inference bằng model đã load sẵn trong bộ nhớ
    # Truyền flattened_samples vào. Class ResNetClassifier sẽ tự động trích xuất frame_data 
    # và chạy theo batch.
    dense_results = resnet_classifier.predict_samples(
        samples=flattened_samples,
        output_json_path=None, # Tạm không ghi trong hàm đó để tự ghi log ở đây
        save_to_disk=False     # Vì ta sẽ tuỳ chỉnh file log dưới đây
    )

    # 3. Sắp xếp lại kết quả theo region và thời gian
    dense_results.sort(key=lambda x: (x.get("region_id", 0), x["time_sec"]))

    # 4. Ghi log
    if save_to_disk and output_json_path:
        output_data = {
            "total_regions": len(dense_regions),
            "total_samples": len(dense_results),
            "results": dense_results
        }
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Đã lưu kết quả Dense Inference vào: {output_json_path}")

    return dense_results