import os
import cv2
import torch
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, img_paths, mask_dir, img_size=640):
        self.img_paths = img_paths
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ----- 이미지 로드 -----
        img_path = self.img_paths[idx]
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # ----- 마스크 로드 -----
        mask_path = os.path.join(self.mask_dir, stem + ".png")
        mask = cv2.imread(mask_path, 0)

        boxes = []
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = (mask > 0).astype("uint8")
            boxes = self.mask_to_boxes(mask)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        if boxes.numel() > 0:
            boxes = boxes.clamp(0.0, 1.0)
        labels = torch.zeros((boxes.shape[0],), dtype=torch.long)

        return img, {
            "boxes": boxes,   # cx, cy, w, h (normalized)
            "labels": labels
        }

    def mask_to_boxes(self, mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = mask.shape
        boxes = []

        for cnt in contours:
            # contour를 감싸는 최소 사각형 (좌상단 x,y + 너비,높이)
            x, y, bw, bh = cv2.boundingRect(cnt)

            # 너무 작은 객체는 노이즈일 가능성이 높아서 제거
            if bw < 3 or bh < 3:
                continue

            # -------------------------------
            # (x, y, w, h) → (cx, cy, w, h) 변환
            # -------------------------------

            # cx, cy:
            # 박스의 중심 좌표를 구함
            # YOLO 계열 모델은 좌상단 기준이 아니라 "중심 좌표"를 사용함
            cx = (x + bw / 2) / w   # 중심 x좌표를 이미지 너비로 나눠 정규화 (0~1)
            cy = (y + bh / 2) / h   # 중심 y좌표를 이미지 높이로 나눠 정규화 (0~1)

            # bw, bh:
            # 박스의 크기를 이미지 크기로 나눠 정규화
            # 이미지 크기가 바뀌어도 동일한 비율 표현 가능
            bw /= w                # 박스 너비 정규화
            bh /= h                # 박스 높이 정규화
            if bw <= 0 or bh <= 0:
                continue

            # YOLO 형식: [center_x, center_y, width, height]
            boxes.append([cx, cy, bw, bh])

        return boxes
