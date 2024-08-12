import os
import json
import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import numpy as np

class VitFeatureExtractor:
    def __init__(self, model_name="vit_large_patch16_224", image_size=224, pretrained=True, freeze=True):
        self.model = create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0 for feature extraction
        self.model.eval()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_features(self, image):
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze(0).numpy()  # Remove batch dimension for output

def load_image_ids(question_paths):
    image_ids = set()
    for question_path in question_paths:
        with open(question_path, 'r') as f:
            questions = json.load(f)
            for question in questions:
                image_ids.add(question['image_id'])
    return image_ids


def extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor):
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)

    for img_name in os.listdir(image_dir):
        img_id = int(img_name.split('_')[-1].split('.')[0])
        if img_id in image_ids:
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            features = vit_extractor.extract_features(image)
            feature_save_path = os.path.join(feature_save_dir, f"{img_id}.npz")
            np.savez_compressed(feature_save_path, x=features)
            print(f"Saved features for image {img_id} to {feature_save_path}")

def train_vqa():
    # OKVQA 질문 파일 경로 설정
    question_paths = [
        "/root/workspace/24s-VQA-MLLM/hankyeol/mcan-vqa/datasets/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
    ]
    
    # 이미지와 저장 경로 설정
    image_dir = "/root/workspace/24s-VQA-MLLM/dataset/coco/images/train2014"
    feature_save_dir = "/root/workspace/24s-VQA-MLLM/features/train2014"
    
    # 이미지 ID 로드
    image_ids = load_image_ids(question_paths)

    # ViT 특징 추출기 초기화
    vit_extractor = VitFeatureExtractor()

    # 특징 추출 및 저장
    extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor)

def val_vqa():
    # OKVQA 질문 파일 경로 설정
    question_paths = [
        "/root/workspace/24s-VQA-MLLM/hankyeol/mcan-vqa/datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
    ]
    
    # 이미지와 저장 경로 설정
    image_dir = "/root/workspace/24s-VQA-MLLM/dataset/coco/images/val2014"
    feature_save_dir = "/root/workspace/24s-VQA-MLLM/features/val2014"
    
    # 이미지 ID 로드
    image_ids = load_image_ids(question_paths)

    # ViT 특징 추출기 초기화
    vit_extractor = VitFeatureExtractor()

    # 특징 추출 및 저장
    extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor)

def test_vqa():
    # OKVQA 질문 파일 경로 설정
    question_paths = [
        "/root/workspace/24s-VQA-MLLM/hankyeol/mcan-vqa/datasets/vqa/v2_OpenEnded_mscoco_test2014_questions.json"
    ]
    
    # 이미지와 저장 경로 설정
    image_dir = "/root/workspace/24s-VQA-MLLM/dataset/coco/images/test2014"
    feature_save_dir = "/root/workspace/24s-VQA-MLLM/features/test2014"
    
    # 이미지 ID 로드
    image_ids = load_image_ids(question_paths)

    # ViT 특징 추출기 초기화
    vit_extractor = VitFeatureExtractor()

    # 특징 추출 및 저장
    extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor)

def train_aokvqa():
    # OKVQA 질문 파일 경로 설정
    question_paths = [
        "/root/workspace/24s-VQA-MLLM/dataset/a-okvqa/aokvqa_v1p0_train.json"
    ]
    
    # 이미지와 저장 경로 설정
    image_dir = "/root/workspace/24s-VQA-MLLM/dataset/a-okvqa/coco/train2017"
    feature_save_dir = "/root/workspace/24s-VQA-MLLM/features/train2017"
    
    # 이미지 ID 로드
    image_ids = load_image_ids(question_paths)

    # ViT 특징 추출기 초기화
    vit_extractor = VitFeatureExtractor()

    # 특징 추출 및 저장
    extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor)

def val_aokvqa():
    # OKVQA 질문 파일 경로 설정
    question_paths = [
        "/root/workspace/24s-VQA-MLLM/dataset/a-okvqa/aokvqa_v1p0_val.json"
    ]
    
    # 이미지와 저장 경로 설정
    image_dir = "/root/workspace/24s-VQA-MLLM/dataset/a-okvqa/coco/val2017"
    feature_save_dir = "/root/workspace/24s-VQA-MLLM/features/val2017"
    
    # 이미지 ID 로드
    image_ids = load_image_ids(question_paths)

    # ViT 특징 추출기 초기화
    vit_extractor = VitFeatureExtractor()

    # 특징 추출 및 저장
    extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor)

def test_aokvqa():
    # OKVQA 질문 파일 경로 설정
    question_paths = [
        "/root/workspace/24s-VQA-MLLM/dataset/a-okvqa/aokvqa_v1p0_test.json"
    ]
    
    # 이미지와 저장 경로 설정
    image_dir = "/root/workspace/24s-VQA-MLLM/dataset/a-okvqa/coco/test2017"
    feature_save_dir = "/root/workspace/24s-VQA-MLLM/features/test2017"
    
    # 이미지 ID 로드
    image_ids = load_image_ids(question_paths)

    # ViT 특징 추출기 초기화
    vit_extractor = VitFeatureExtractor()

    # 특징 추출 및 저장
    extract_and_save_features(image_dir, image_ids, feature_save_dir, vit_extractor)

if __name__ == "__main__":
    val_aokvqa()