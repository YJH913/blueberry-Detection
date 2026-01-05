import os
import cv2
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A  # type: ignore
except Exception:
    A = None


class MaskDataset(Dataset):
    def __init__(self, img_paths, mask_dir, img_size=640, augment: bool = False):
        self.img_paths = img_paths
        self.mask_dir = mask_dir
        self.img_size = img_size
        # ResNet-18 from scratch에도 정상 동작하는 표준 RGB 정규화
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # 논문에서 사용한 밝기/대비, 비, 그림자, 눈, ISO 노이즈, 태양광 플레어, 안개 등의 증강
        # 지오메트리를 바꾸지 않는 photometric 변환만 적용하므로 bbox에는 영향이 없음.
        self.augment = augment and (A is not None)
        if self.augment:
            self.aug = A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomRain(p=0.2),
                    A.RandomShadow(p=0.2),
                    A.RandomSnow(p=0.2),
                    A.ISONoise(p=0.2),
                    A.RandomSunFlare(p=0.2),
                    A.RandomFog(p=0.2),
                ]
            )
        else:
            self.aug = None

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

        # photometric augmentation (train split만)
        if self.aug is not None:
            out = self.aug(image=img)
            img = out["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        img = (img - self.mean) / self.std

        # ----- 마스크 로드 -----
        mask_path = os.path.join(self.mask_dir, stem + ".png")
        mask = cv2.imread(mask_path, 0)

        boxes = []
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            boxes = self.mask_to_boxes(mask)

        boxes = torch.tensor(boxes, dtype=torch.float32)
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

            # YOLO 형식: [center_x, center_y, width, height]
            boxes.append([cx, cy, bw, bh])

        return boxes
