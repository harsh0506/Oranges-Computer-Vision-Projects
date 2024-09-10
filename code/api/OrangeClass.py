import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
import json
import time


class OrangeDetectorAndClassifier_v2:
    def __init__(
        self,
        yolo_model_path: str,
        classification_model_type: str,
        class_names,
        classification_model_path=None,
        config_path=None,
        model_path=None,
        confidence_threshold=0.5,
        device=None,
    ):
        self.yolo_model = YOLO(yolo_model_path)
        self.class_names = class_names
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.classification_model_type = classification_model_type.lower()

        # Load classification model based on the specified type
        self.classification_model_type = classification_model_type.lower()
        if self.classification_model_type == "efficientnet":
            self.classification_model = self._load_efficientnet(class_names)
        elif self.classification_model_type == "vit":
            self.classification_model, self.feature_extractor = self._load_vit(
                class_names, classification_model_path
            )
        else:
            raise ValueError("Unsupported model type. Use 'efficientnet' or 'vit'.")

    def _load_efficientnet(self, class_names):
        model = models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, len(class_names)
        )
        model.eval()
        model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return model

    def _load_vit(self, class_names, model_path):
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=len(class_names)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        return model, feature_extractor

    def detect_oranges(self, image_path):
        start_time = time.time()
        results = self.yolo_model([image_path])
        detection_time = time.time() - start_time

        b_boxes = []
        counter = 0
        
        for result in results:
            boxes = result.boxes  # Bounding box outputs
            filtered_boxes = boxes[
                boxes.conf > self.confidence_threshold
            ]  # Filter boxes with confidence > threshold
            counter += 1
            b_boxes.append(filtered_boxes)
            
            
        return b_boxes, detection_time, len(b_boxes[0].conf.tolist())

    def crop_image(self, img, b_box):
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = b_box.xyxy[0].tolist()
        crop_img = img.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        return crop_img

    def classify_orange(self, cropped_img, classification_model_type=None):
        model_type = classification_model_type if classification_model_type else self.classification_model_type
        if model_type == "efficientnet":
            image = self.transform(cropped_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classification_model(image)
                predicted = torch.max(outputs, 1)[1]
        elif model_type == "vit":
            inputs = self.feature_extractor(images=cropped_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                predicted = outputs.logits.argmax(-1)

        
        return self.class_names[predicted.item()]
    
    def _save_results(self, output_json_path, result_json):
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as json_file:
                existing_data = json.load(json_file)
            if isinstance(existing_data, list):
                existing_data.append(result_json)
            else:
                existing_data = [existing_data, result_json]
        else:
            existing_data = [result_json]

        with open(output_json_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    def process_image(self, image_path, output_json_path):
        b_boxes, detection_time, orange_count  = self.detect_oranges(image_path)
        img = Image.open(image_path)
        orange_predictions = []
        classification_time = 0

        for b_box in b_boxes:
            cropped_img = self.crop_image(img, b_box)
            start_time = time.time()
            orange_type = self.classify_orange(cropped_img)
            classification_time += time.time() - start_time

            orange_predictions.append(
                {
                    "cropped_orange_coordinates": b_box.xyxy[0].tolist(),
                    "orange_type": orange_type,
                }
            )

        result_json = {
            "original_image": image_path,
            "count_of_detected_oranges": len(b_boxes),
            "oranges_predicted_classification": orange_predictions,
            "time_for_detection": detection_time,
            "total_time_for_classification": classification_time,
        }

        self._save_results(output_json_path, result_json)
        return result_json
