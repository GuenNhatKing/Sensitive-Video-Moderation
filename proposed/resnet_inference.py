import json
import logging
import os
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ResNet-50 FCN
class ResNet50FCN(nn.Module):
    """
    ResNet-50 Fully Convolutional.
    Backbone: ResNet-50 conv layers
    Classifier: 1x1 Conv
    Output: Global Average Pooling -> logits [B, num_classes]
    """

    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.classifier = nn.Conv2d(
            in_channels=2048,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, start_dim=1)
        return x


# ResNet Classifier
class ResNetClassifier:

    def __init__(
        self,
        model_path,
        batch_size=21,
        device=None
    ):
        self.device = (
            device
            or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.model_path = model_path

        # ImageNet normalization
        self.imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(3, 1, 1)

        self.imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(3, 1, 1)

        self.target_width = 640
        self.target_height = 360

        logging.info("Khởi tạo ResNet-50-FCN | Spatial Grid 360x640 (stride 32: 12x20 = 240 vùng đặc trưng)")

        self._load_model()

    def _load_model(self):
        logging.info(f"Đang tải checkpoint từ {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.classes = checkpoint["classes"]
        self.class_to_idx = checkpoint["class_to_idx"]
        self.adult_idx = self.class_to_idx["adult"]
        num_classes = len(self.classes)

        self.model = ResNet50FCN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info(f"Load model thành công. Classes: {self.classes}")

    def _get_frame_bgr(self, sample):
        frame_bgr = sample.get("frame_data")
        if frame_bgr is None:
            image_path = sample.get("image_path")
            if image_path is None or not os.path.exists(image_path):
                raise FileNotFoundError(f"Không tìm thấy image: {image_path}")
            frame_bgr = cv2.imread(image_path)

        if frame_bgr is None:
            raise ValueError("Không thể đọc frame.")
        return frame_bgr

    def crop_center_360x640(self, frame_bgr):
        """
        Thực hiện CenterCrop với kích thước 360x640 (Height=360, Width=640).
        Nếu ảnh nhỏ hơn 360x640, lấy vùng trung tâm 70% và resize về 360x640.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        if orig_h >= self.target_height and orig_w >= self.target_width:
            y1 = (orig_h - self.target_height) // 2
            x1 = (orig_w - self.target_width) // 2
            crop = frame_bgr[y1:y1 + self.target_height, x1:x1 + self.target_width]
        else:
            crop_h = max(1, int(orig_h * 0.7))
            crop_w = max(1, int(orig_w * 0.7))
            y1 = (orig_h - crop_h) // 2
            x1 = (orig_w - crop_w) // 2
            crop = frame_bgr[y1:y1 + crop_h, x1:x1 + crop_w]
            crop = cv2.resize(crop, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        return crop

    def _preprocess_frame(self, frame_bgr, is_center_crop=False):
        """
        Chuẩn hóa frame đưa vào ResNet-50 FCN:
        - Mặc định: resize trực tiếp về 360x640 (Height=360, Width=640)
        - Nếu is_center_crop=True: thực hiện CenterCrop 360x640
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        if is_center_crop:
            img = self.crop_center_360x640(frame_bgr)
        else:
            if (orig_w, orig_h) != (self.target_width, self.target_height):
                interp = cv2.INTER_AREA if (orig_w > self.target_width and orig_h > self.target_height) else cv2.INTER_LINEAR
                img = cv2.resize(frame_bgr, (self.target_width, self.target_height), interpolation=interp)
            else:
                img = frame_bgr

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Chuẩn hóa Tensor [0, 1] và trừ ImageNet mean/std
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.imagenet_mean) / self.imagenet_std

        return tensor, "custom", orig_w, orig_h, self.target_width, self.target_height

    def _predict_batch(self, tensors):
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def predict_samples(
        self,
        samples,
        output_json_path=None,
        save_to_disk=True
    ):
        if not samples:
            return []

        logging.info(f"Bắt đầu tiền xử lí và inference {len(samples)} frames...")

        # Gom nhóm theo target_size để đảm bảo batching hợp lệ
        # groups: {(target_w, target_h): [(orig_index, tensor, sample, meta_info)]}
        groups = defaultdict(list)

        for idx, sample in enumerate(samples):
            frame_bgr = self._get_frame_bgr(sample)
            (
                tensor,
                ratio_type,
                orig_w,
                orig_h,
                target_w,
                target_h
            ) = self._preprocess_frame(frame_bgr)

            meta = {
                "ratio_type": ratio_type,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "target_w": target_w,
                "target_h": target_h
            }
            groups[(target_w, target_h)].append((idx, tensor, sample, meta))

        # Lưu kết quả theo index gốc để giữ nguyên thứ tự
        results_by_index = {}

        # Chạy inference theo từng nhóm kích thước
        for (w, h), items in groups.items():
            logging.info(f"Đang inference nhóm kích thước {w}x{h} ({items[0][3]['ratio_type']}): {len(items)} frames")

            for start in range(0, len(items), self.batch_size):
                batch_items = items[start:start + self.batch_size]
                batch_tensors = [it[1] for it in batch_items]

                probabilities = self._predict_batch(batch_tensors)

                for i, (orig_idx, _, sample, meta) in enumerate(batch_items):
                    probs = probabilities[i]
                    predicted_idx = int(torch.argmax(probs).item())
                    adult_probability = float(probs[self.adult_idx].item())

                    result = {
                        "region_id": sample.get("region_id"),
                        "shot_id": sample.get("shot_id"),
                        "sample_id": sample.get("sample_id"),
                        "frame_index": sample.get("frame_index"),
                        "time_sec": sample.get("time_sec"),
                        "image_path": sample.get("image_path", "in_memory"),
                        "aspect_ratio_type": meta["ratio_type"],
                        "original_size": f"{meta['orig_w']}x{meta['orig_h']}",
                        "input_size": f"{meta['target_w']}x{meta['target_h']}",
                        "predicted_class": self.classes[predicted_idx],
                        "adult_probability": round(adult_probability, 6),
                    }
                    results_by_index[orig_idx] = result

        # Khôi phục đúng thứ tự ban đầu của danh sách samples
        results = [results_by_index[i] for i in range(len(samples))]

        # Ghi file JSON nếu cần
        if save_to_disk and output_json_path:
            output_dir = os.path.dirname(output_json_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

            logging.info(f"Đã lưu kết quả: {output_json_path}")

        logging.info(f"Hoàn thành inference: {len(results)} frames.")
        return results

    def predict_frame_data(self, frame_bgr, is_center_crop=False):
        """
        Dự đoán xác suất Adult cho 1 frame BGR trực tiếp trên RAM.
        Nếu is_center_crop=True: cắt vùng trung tâm 360x640 trước khi đưa vào mô hình.
        """
        tensor, _, _, _, _, _ = self._preprocess_frame(frame_bgr, is_center_crop=is_center_crop)
        probs = self._predict_batch([tensor])
        adult_prob = float(probs[0, self.adult_idx].item())
        return adult_prob

    def predict_frame_batch(self, frames_bgr, is_center_crop=False):
        """
        Dự đoán xác suất Adult theo batch cho danh sách frame BGR.
        """
        if not frames_bgr:
            return []
        tensors = [self._preprocess_frame(f, is_center_crop=is_center_crop)[0] for f in frames_bgr]
        probs = self._predict_batch(tensors)
        return [float(p[self.adult_idx].item()) for p in probs]