import logging
import cv2
import numpy as np


class DiversitySpatialSampler:
    r"""
    THAY ĐỔI 3: Phóng đại Không gian (360x640) & Chọn 4 Frame Đa dạng Cực đại
    (Thay thế Dense Sampling 10 FPS ở 180x320)
    - Grid mới 360x640: ceil(360/32) x ceil(640/32) = 12 x 20 = 240 vùng đặc trưng.
    - Cửa sổ Omega(t_0):
        t_start = max(t_0 - 1.0s, start_shot)
        t_end   = min(t_0 + 1.0s, end_shot)
        Omega(t_0) = { f_t | t thuộc [t_start, t_end] }
    - Max-Min Diversity (Farthest Point Sampling):
        Đưa frames về ảnh xám 32x32, chuẩn hóa vector đơn vị v_t (||v_t||_2 = 1).
        D_1 = { f_{t_0} }
        Tại bước k (k=1..4):
            f_next = argmax_{f_c in Omega \ D_k} ( min_{f_s in D_k} ||v_c - v_s||_2 )
            D_{k+1} = D_k U { f_next }
    """

    def __init__(self, time_window_radius=1.0, sampling_fps=10.0, target_num_frames=5):
        self.time_window_radius = time_window_radius
        self.sampling_fps = sampling_fps
        self.target_num_frames = target_num_frames
        # Cache bộ nhớ đệm trong RAM cho frame và vector đặc trưng để tránh decode lặp lại
        # và tự động gộp các cửa sổ thời gian chồng lấn (overlapping windows)
        self._frame_cache = {}
        self._vector_cache = {}
        self._current_cap_pos = -1

    def clear_cache(self):
        """Giải phóng bộ đệm RAM khi chuyển sang video mới."""
        self._frame_cache.clear()
        self._vector_cache.clear()
        self._current_cap_pos = -1

    def _frame_to_unit_vector(self, frame_bgr):
        """
        Chuyển frame BGR sang ảnh xám 32x32, làm phẳng thành vector 1024 chiều,
        chuẩn hóa L2 thành vector đơn vị: ||v||_2 = 1.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        vec = small.astype(np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec = np.zeros_like(vec)
        return vec

    def extract_diverse_frames(self, cap, fps, anchor, video_duration_sec):
        """
        Trích xuất tập D* gồm 5 frames cho anchor t_0:
        - Anchor t_0 và 4 frames bổ sung có sự khác biệt thị giác cực đại.
        """
        t_0 = anchor["time_sec"]
        start_shot = anchor.get("start_shot", 0.0)
        end_shot = anchor.get("end_shot", video_duration_sec)

        # 1. Cửa sổ thời gian lân cận Omega(t_0)
        t_start = max(t_0 - self.time_window_radius, start_shot, 0.0)
        t_end = min(t_0 + self.time_window_radius, end_shot, video_duration_sec)

        if t_end <= t_start:
            t_end = min(t_start + 0.2, video_duration_sec)

        # 2. Lấy mẫu dày (Dense Sampling 10 FPS) trong khoảng [t_start, t_end]
        dense_interval = 1.0 / self.sampling_fps
        candidate_timestamps = list(np.arange(t_start, t_end + 1e-4, dense_interval))
        if not candidate_timestamps or candidate_timestamps[0] != t_start:
            candidate_timestamps.insert(0, t_start)
        if candidate_timestamps[-1] != t_end:
            candidate_timestamps.append(t_end)

        # Đảm bảo mốc t_0 có mặt chính xác trong danh sách
        candidate_timestamps.append(t_0)
        candidate_timestamps = sorted(list(set(np.round(candidate_timestamps, 4))))

        # 3. Đọc dữ liệu các frame trong Omega(t_0) và tính vector đơn vị v_t
        # Tối ưu hóa: Tránh seek lặp lại trên H.264 GOP, dùng cache RAM và đọc tuần tự tiến lên (Sequential Read)
        omega = []
        anchor_frame_data = anchor.get("frame_data")
        anchor_f_idx = anchor.get("frame_index")

        if anchor_f_idx is not None and anchor_frame_data is not None:
            self._frame_cache[anchor_f_idx] = anchor_frame_data
            if anchor_f_idx not in self._vector_cache:
                self._vector_cache[anchor_f_idx] = self._frame_to_unit_vector(anchor_frame_data)

        target_f_indices = [int(round(ts * fps)) for ts in candidate_timestamps]
        missing_indices = sorted(list(set(idx for idx in target_f_indices if idx not in self._frame_cache)))

        if missing_indices:
            for idx in missing_indices:
                if 0 <= self._current_cap_pos <= idx and (idx - self._current_cap_pos) <= 30:
                    while self._current_cap_pos < idx:
                        ret, _ = cap.read()
                        if not ret:
                            break
                        self._current_cap_pos += 1
                    ret, frame = cap.read()
                    if ret:
                        self._current_cap_pos += 1
                        self._frame_cache[idx] = frame
                        self._vector_cache[idx] = self._frame_to_unit_vector(frame)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        self._current_cap_pos = idx + 1
                        self._frame_cache[idx] = frame
                        self._vector_cache[idx] = self._frame_to_unit_vector(frame)

        for f_idx in target_f_indices:
            if f_idx not in self._frame_cache:
                continue
            frame = self._frame_cache[f_idx]
            v_t = self._vector_cache.get(f_idx)
            if v_t is None:
                v_t = self._frame_to_unit_vector(frame)
                self._vector_cache[f_idx] = v_t

            is_anchor_frame = bool(f_idx == anchor_f_idx)
            omega.append({
                "frame_index": f_idx,
                "time_sec": round(f_idx / fps, 4),
                "is_anchor": is_anchor_frame,
                "vec": v_t,
                "frame_data": frame
            })

        if not omega:
            # Fallback nếu không đọc được frame nào
            if anchor_frame_data is not None:
                v_t = self._frame_to_unit_vector(anchor_frame_data)
                omega = [{
                    "frame_index": anchor.get("frame_index", 0),
                    "time_sec": t_0,
                    "is_anchor": True,
                    "vec": v_t,
                    "frame_data": anchor_frame_data
                }]
            else:
                return []

        # Tìm frame tâm điểm f_{t_0}
        f_t0 = None
        for item in omega:
            if item["is_anchor"]:
                f_t0 = item
                break
        if f_t0 is None:
            # Lấy frame có thời gian gần t_0 nhất
            f_t0 = min(omega, key=lambda it: abs(it["time_sec"] - t_0))
            f_t0["is_anchor"] = True

        # 4. Greedy Max-Min Diversity (Farthest Point Sampling)
        # Khởi tạo D_1 = { f_{t_0} }
        D = [f_t0]
        selected_indices = {f_t0["frame_index"]}

        # Số frame cần chọn thêm (tối đa 4 frames để |D*| = 5)
        k_needed = min(self.target_num_frames - 1, len(omega) - 1)

        for _ in range(k_needed):
            best_cand = None
            best_min_dist = -1.0

            for cand in omega:
                if cand["frame_index"] in selected_indices:
                    continue

                # min_{f_s in D_k} sqrt( sum( (v_c - v_s)^2 ) )
                distances = [
                    float(np.linalg.norm(cand["vec"] - s["vec"]))
                    for s in D
                ]
                min_dist = min(distances) if distances else 0.0

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_cand = cand

            if best_cand is not None:
                D.append(best_cand)
                selected_indices.add(best_cand["frame_index"])

        # Trả về tập D* được sắp xếp theo trình tự thời gian
        # và đánh dấu rõ đâu là frame tâm điểm f_{t_0}
        return D
