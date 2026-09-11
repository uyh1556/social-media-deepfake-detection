import cv2
import os
import glob
import json
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

# ==========================================
# [설정 영역]
# ==========================================

# 프로젝트 루트 기준:
# ~/Desktop/Projects/deepfake_project
PROJECT_ROOT = "./ffpp_data"

# FaceForensics++ raw video 경로
VIDEO_DIRS = {
    # "original": os.path.join(PROJECT_ROOT, "original_sequences", "youtube", "raw", "videos"),
    # "Deepfakes": os.path.join(PROJECT_ROOT, "manipulated_sequences", "Deepfakes", "raw", "videos"),
    "Face2Face": os.path.join(PROJECT_ROOT, "manipulated_sequences", "Face2Face", "raw", "videos"),
}

# 결과 저장 루트
OUTPUT_ROOT = "./source_images_raw"

# 추출 옵션
NUM_FRAMES_PER_VIDEO = 10
CONFIDENCE_THRESHOLD = 0.90

# 원본 해상도 유지
RESIZE_SHAPE = None

# 얼굴 오탐 필터 옵션
MIN_FACE_AREA_RATIO = 0.02
MAX_FACE_AREA_RATIO = 0.60
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
MAX_CENTER_Y_RATIO = 0.75

# Resume 상태 저장 디렉토리
STATE_DIR = "./extract_state"

# ==========================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_frames_uniformly(total_frames, num_samples):
    if total_frames <= 0:
        return []
    if total_frames < num_samples:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()


def get_output_subdir(dataset_name):
    """
    저장 폴더 구조:
    source_images_raw/
        original/
        Deepfakes/
        Face2Face/
    """
    return os.path.join(OUTPUT_ROOT, dataset_name)


def get_state_file(dataset_name):
    ensure_dir(STATE_DIR)
    return os.path.join(STATE_DIR, f"{dataset_name}_processed.json")


def load_processed_set(dataset_name):
    state_file = get_state_file(dataset_name)
    if not os.path.exists(state_file):
        return set()

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data)
    except Exception:
        return set()


def save_processed_set(dataset_name, processed_set):
    state_file = get_state_file(dataset_name)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_set)), f, ensure_ascii=False, indent=2)


def is_valid_face_detection(result, frame_shape):
    """
    MTCNN detection 결과가 유효한 얼굴인지 판단
    """
    confidence = result.get("confidence", 0.0)
    if confidence < CONFIDENCE_THRESHOLD:
        return False

    box = result.get("box", None)
    if box is None or len(box) != 4:
        return False

    x, y, w, h = box
    img_h, img_w = frame_shape[:2]

    # 비정상 bbox 방지
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > img_w or y + h > img_h:
        return False

    img_area = img_w * img_h
    face_area = w * h
    face_area_ratio = face_area / (img_area + 1e-6)

    if face_area_ratio < MIN_FACE_AREA_RATIO or face_area_ratio > MAX_FACE_AREA_RATIO:
        return False

    aspect_ratio = h / (w + 1e-6)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False

    center_y = y + h / 2.0
    if center_y > img_h * MAX_CENTER_Y_RATIO:
        return False

    return True


def process_video(video_path, save_dir, detector):
    """
    영상 1개 처리:
    - uniform sampling으로 최대 NUM_FRAMES_PER_VIDEO개의 후보 프레임 선택
    - 유효한 얼굴이 하나라도 있으면 full frame PNG 저장
    - 저장 개수와 관계없이 '영상 처리 완료' 자체를 resume state에 기록
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[WARN] Invalid frame count: {video_path}")
        cap.release()
        return False

    target_indices = get_frames_uniformly(total_frames, NUM_FRAMES_PER_VIDEO)

    current_frame = 0
    target_idx_ptr = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if target_idx_ptr < len(target_indices) and current_frame == target_indices[target_idx_ptr]:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                results = detector.detect_faces(rgb_frame)
            except Exception as e:
                print(f"[WARN] Face detection failed: {video_name}, frame={current_frame}, error={e}")
                results = []

            has_valid_face = False
            for result in results:
                if is_valid_face_detection(result, frame.shape):
                    has_valid_face = True
                    break

            if has_valid_face:
                save_filename = f"{video_name}_frame{current_frame}.png"
                save_path = os.path.join(save_dir, save_filename)

                if RESIZE_SHAPE:
                    final_img = cv2.resize(frame, RESIZE_SHAPE)
                else:
                    final_img = frame

                cv2.imwrite(save_path, final_img)
                saved_count += 1

            target_idx_ptr += 1

        current_frame += 1
        if target_idx_ptr >= len(target_indices):
            break

    cap.release()
    return True


def process_dataset(dataset_name, detector):
    video_dir = VIDEO_DIRS[dataset_name]
    save_dir = get_output_subdir(dataset_name)
    ensure_dir(save_dir)

    if not os.path.exists(video_dir):
        print(f"[WARN] Video directory does not exist: {video_dir}")
        return

    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    processed_set = load_processed_set(dataset_name)

    print(f"\n[{dataset_name}] video dir: {video_dir}")
    print(f"[{dataset_name}] output dir: {save_dir}")
    print(f"[{dataset_name}] found videos: {len(video_paths)}")
    print(f"[{dataset_name}] already processed: {len(processed_set)}")

    for video_path in tqdm(video_paths, desc=f"Processing {dataset_name}"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 개선된 resume:
        # "파일이 하나라도 있으면 skip"이 아니라
        # "정상 처리 완료된 영상 이름" 기준으로 skip
        if video_name in processed_set:
            continue

        success = process_video(video_path, save_dir, detector)

        if success:
            processed_set.add(video_name)
            save_processed_set(dataset_name, processed_set)

    num_saved = len(glob.glob(os.path.join(save_dir, "*.png")))
    print(f"[{dataset_name}] saved PNGs: {num_saved}")


def main():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(STATE_DIR)

    print(f"Project root: {os.path.abspath(PROJECT_ROOT)}")
    print(f"Output root : {os.path.abspath(OUTPUT_ROOT)}")
    print("Loading MTCNN model...")
    detector = MTCNN()

    # for dataset_name in ["original", "Deepfakes", "Face2Face"]:
    #     process_dataset(dataset_name, detector)
    for dataset_name in ["Face2Face"]:
        process_dataset(dataset_name, detector)

    print("\nAll done.")
    # for dataset_name in ["original", "Deepfakes", "Face2Face"]:
    #     out_dir = get_output_subdir(dataset_name)
    #     count = len(glob.glob(os.path.join(out_dir, "*.png")))
    #     print(f"{dataset_name}: {count} PNG files")
    for dataset_name in ["Face2Face"]:
        out_dir = get_output_subdir(dataset_name)
        count = len(glob.glob(os.path.join(out_dir, "*.png")))
        print(f"{dataset_name}: {count} PNG files")

if __name__ == "__main__":
    main()