# Orange Computer Vision Pipeline

## Project Overview

This project is designed to explore state-of-the-art (SOTA) computer vision algorithms and techniques, particularly when working with small datasets. Given my interest in oranges, I've chosen to develop a full-stack project focusing on various computer vision tasks related to oranges, including:

- Orange detection and localization
- Classification into 5 species
- Disease detection and segmentation (across 7 diseases)

I have already implemented some techniques and plan to explore more as the project progresses.

## Motivation

As a second-year master's student, my goal is to delve deep into creating a highly scalable and robust computer vision product. This project is a means to demonstrate my understanding of four fundamental tasks in computer vision:

1. **Classification**
2. **Object Detection**
3. **Segmentation**
4. **Image Creation/Captioning**

The dataset for this study was self-created by capturing images from local fruit shops and markets using my mobile camera. This project also serves to showcase my capabilities in orchestrating and deploying robust computer vision solutions.

## Dataset

### Size and Composition

- **Classification Dataset**: 
  - 5 classes
  - 6-8 high-quality images per class
- **Segmentation Dataset**:
  - COCO and YOLO formats
  - ~30 images
  - 7-8 classes

### Creation and Annotation Tools

- **Image Capture**: Mobile camera, sourced from local fruit markets
- **Annotation Tool**: [MakeSense.ai](https://www.makesense.ai/)

## FastAPI Service Development

### Reason for Using FastAPI

FastAPI was chosen for its speed, ease of use, and scalability. The logic for deep learning tasks is encapsulated in a class, offering four primary methods:

1. **count_oranges**: Count the number of oranges in an image.
2. **classify_orange**: Classify an orange into one of the five species.
3. **detect_disease**: Detect and identify diseases in the oranges.
4. **process_pipeline**: A complete pipeline to detect, crop, classify, segment, and report.

### API Routes

- **GET** `/web_page`: Serve a web page (if applicable)
- **POST** `/count`: Count oranges in an image
- **POST** `/classify`: Classify an orange species
- **POST** `/segment`: Segment the orange and detect diseases
- **POST** `/process`: Execute the full pipeline

### Response Format

All responses are returned in a consistent JSON schema. Images are stored in Google Cloud Storage.

## Folder Structure

```plaintext
code/
    ├── api/  # API code for CV tasks
    └── Model training/  # Notebooks for model fine-tuning
dataset/
    ├── classification_dataset/
    └── segmentation_dataset/
models/  # Trained model files
Dockerfile
README.md
requirements.txt
```

## Libraries Used

- `pytorch`
- `ultralytics`
- `transformers`
- `pillow`
- `detectron2`
- `matplotlib`
- `torchvision`
- `fastapi`

## Algorithms and Techniques

### Image Classification
- Transfer Learning: ResNet, ImageNet, MobileNet, RegNet, EfficientNet, ViT

### Image Segmentation
- YOLO-Seg, Detectron-2, U-Net, DeepLabV3+

### Object Detection
- YOLOv8

## Results

- **Classification Evaluation**: To be added soon.
- **Segmentation Evaluation Score**: To be added soon.

## How to Use

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Build the Docker image:
    ```bash
    docker build -t orange_computer_vision_pipeline_v1 .
    ```
3. Run the Docker container:
    ```bash
    docker run -it -p 8000:8000 orange_computer_vision_pipeline_v1
    ```

## Deployment Plans

Deployment is planned on either Google Cloud or AWS.

## To-Do List

- Implement image creation using diffusion models
- Integrate advanced SOTA classification and segmentation models
- Add evaluation reports
- Optimize image sizes
- Conduct API testing
