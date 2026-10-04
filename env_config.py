import os
from pathlib import Path

# Xác định thư mục src và project root
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

def load_env(env_path=None):
    """
    Tự động tìm và nạp các biến môi trường từ file .env vào os.environ.
    Hỗ trợ tìm kiếm tại cwd, src, hoặc project root.
    """
    if env_path is None:
        candidates = [
            Path.cwd() / ".env",
            SRC_DIR / ".env",
            PROJECT_ROOT / ".env"
        ]
        for cand in candidates:
            if cand.is_file():
                env_path = cand
                break

    if env_path and os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

def resolve_path(path_str):
    """
    Giải quyết đường dẫn một cách thông minh:
    - Nếu là đường dẫn tuyệt đối: giữ nguyên nếu tồn tại
    - Kiểm tra lần lượt: cwd, SRC_DIR, PROJECT_ROOT
    - Nếu là đường dẫn tương đối bắt đầu bằng ../ : ưu tiên theo SRC_DIR
    """
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p.resolve())
    
    # Nếu đường dẫn tồn tại ngay tại cwd
    if (Path.cwd() / p).exists():
        return str((Path.cwd() / p).resolve())
        
    # Nếu đường dẫn tồn tại trong SRC_DIR
    if (SRC_DIR / p).exists():
        return str((SRC_DIR / p).resolve())
        
    # Nếu đường dẫn tồn tại trong PROJECT_ROOT
    if (PROJECT_ROOT / p).exists():
        return str((PROJECT_ROOT / p).resolve())

    # Nếu bắt đầu bằng ../ (thường là chạy từ src)
    if str(path_str).startswith(".."):
        return str((SRC_DIR / p).resolve())
        
    return str((SRC_DIR / p).resolve())

def get_env(key, default=None, resolve=False):
    """Lấy giá trị cấu hình từ biến môi trường (hoặc .env)"""
    load_env()
    val = os.environ.get(key, default)
    if resolve and val:
        return resolve_path(val)
    return val



def resolve_video_path(raw_video_path, json_path=None):
    """
    Xác định vị trí thực tế của video:
    1. Kiểm tra trực tiếp đường dẫn lưu trong JSON.
    2. Nếu không tìm thấy, thử tương đối so với vị trí file JSON.
    3. Thử tìm theo tên file trong các thư mục Test phổ biến.
    """
    if not raw_video_path:
        return None
    p = Path(raw_video_path)
    if p.exists():
        return str(p.resolve())
    
    if json_path:
        cand_json = (Path(json_path).resolve().parent / raw_video_path)
        if cand_json.exists():
            return str(cand_json.resolve())
            
    # Tìm theo tên file trong các thư mục Test phổ biến
    video_name = p.name
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "Test",
        PROJECT_ROOT / "Test",
        SRC_DIR / "Test",
        Path.cwd().parent / "Test"
    ]
    for d in search_dirs:
        cand = d / video_name
        if cand.exists():
            return str(cand.resolve())
            
    return raw_video_path
