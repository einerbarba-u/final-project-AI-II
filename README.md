# Sports Analytics using Deep Learning -- AI II Final Project

Universidad de Monterrey - Escuela de Ingenieria y Tecnologias
Course: Artificial Intelligence I.
Lecturer: Dr. Andres Hernandez Gutierrez
Name: Einer Barba Abdala (595839)
Name: Renata Garcia Morales (612194)

---

## Project Description

This project implements a deep learning computer vision pipeline to detect soccer players
and the soccer ball from aerial drone footage captured over the UDEM campus soccer field
The details of the footage consists of a DJI Mavic Pro. 1280x720 of 30 fps and the lenght is 2 min 8 sec.

A yolo26n.pt was fine-tuned via transfer learning on a manually annotated
subset of 575 frames labelled with Roboflow. The pipeline processes every frame of the full
video and produces a real-time annotated output that has the next visual analytics:

- Player count per frame
- Players in the left half and right half of the field
- Players inside the left and right penalty areas
- Players detected outside the field boundary
- Ball position: left half, right half, or not detected

The system automatically estimates the field boundaries, the halfway line and the penalty areas in each frame using HSV green segmentation, and it smooths the results over time with exponential blending to keep them consistent.

---

## Structure of the repository 

    final-project-AI-II/
    |
    |-- Initial_implementation_of_a_DL_Pipeline.ipynb   
    |-- yolo26n.pt                                      
    |-- runs/
    |   |-- detect/
    |       |-- train-4/weights/best.pt                  
    |-- split/
    |   |-- train/images & labels   # 387 training samples
    |   |-- val/images & labels     # 83 validation samples
    |   |-- test/images & labels    # 83 test samples
    |   |-- data.yaml
    |-- predictions/
    |   |-- live_analysis.mp4       # Annotated output video
    |-- requirements.txt
    |-- README.md


---

## Requirements

- Python version 3.10 to 3.12
- A CUDA capable GPU is recommended, tested on NVIDIA RTX 2080 SUPER with 8 GB of memory
- Place the source video in the project root
- Place the file SoccerField.yolo26.zip in the project root if you want to re‑run training

---

## How to Run

1. First, clone the repository.

    git clone https://github.com/einerbarba-u/final-project-AI-II.git
    cd final-project-AI-II

2. Then, create a virtual environment.

    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate

3. After this, install the dependencies.

    pip install -r requirements.txt

   For GPU acceleration, install the CUDA-enabled PyTorch build before running the notebook:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

4. Continue by adding the required files to the project root

    final-project-AI-II/
    |-- 2023_05_05_15_02_22-players-and-ball-detection.mp4
    |-- SoccerField.yolo26.zip        

5. Open and run the notebook which is Initial_implementation_of_a_DL_Pipeline.ipynb

    The notebook is divided into the following stages:

    Stage 1. Data preparation: Extract 405 frames using fixed‑interval sampling in two passes
    Stage 2. Data labeling: Load the Roboflow zip and pull out the annotated frames
    Stage 3. Dataset split: Divide frames into 70 percent training, 15 percent validation, 15 percent testing
    Stage 4. Model training: Fine‑tune yolo26n.pt for 100 epochs
    Stage 5. Validation: Test the best checkpoint on the evaluation set
    Stage 6. Video prediction: Run the pipeline on the full video and save the output as live_analysis.mp4

    In order to skip re-training and use the already fine-tuned weights, you can go directly to the Predictions section. Make sure runs/detect/train-4/weights/best.pt is present by already having it committed to the repo.

---

## Expected Output

    Each frame of the video shows:

    - Red bounding boxes: players detected, with confidence and zone  
    - Cyan dot: ball position  
    - White polygon: field boundary estimated with HSV segmentation  
    - White vertical line: halfway line  

    The following is an example of the analytics panel:

    Total players: 13  
    Players in the left half: 8  
    Players in the right half: 4  
    Ball position: left half  
    Players in the left penalty area: 1  
    Players in the right penalty area: 0  
    Players outside the field: 0  

---

## Model Details

    Architecture: YOLO26n  
    Pre-trained on: COCO (yolo26n.pt)  
    Fine-tuning epochs: 100  
    Image size: 1280 px  
    Batch size: 8  
    Loss weights: box 12.0, cls 3.0, dfl 2.0  
    Framework: Ultralytics 8.4.47 with PyTorch 2.6.0+cu124  

    Classes from Roboflow export:  
    0 ball  
    1 player  

    Test set metrics that are to be filled after running model.val:  
    Precision for player, Precision for ball, Recall for player, Recall for ball, F1-score for player, F1-score for ball, mAP@50 for player, mAP@50 for ball.

---

## Dataset

    Source: Aerial video with 3,840 frames at 1280x720 and 30 fps  
    Sampling strategy: First pass every 20 frames for 192 frames, second pass every 4 frames until reaching 575 in total  
    Annotation tool: Roboflow at https://roboflow.com  
    Format: YOLO txt, one label file per image  
    Classes: ball as 0, player as 1  
    Total annotated: 575 frames after the Roboflow export  
    Split: 387 training, 83 validation, 83 testing  


---

## References

    GeeksforGeeks. (2025, January 23). Extract images from video in Python. GeeksforGeeks. https://www.geeksforgeeks.org/python/extract-images-from-video-in-python/

    Kandhare, M., & Gisselbrecht, T. (2024). An empirical comparison of video frame sampling methods for multi-modal RAG retrieval. arXiv. https://arxiv.org/html/2408.03340v1

    Roboflow. (2022, June 15). Tips for how to label images. Roboflow Blog. https://blog.roboflow.com/tips-for-how-to-label-images/

    Ultralytics. (n.d.). COCO dataset. Ultralytics Docs. https://docs.ultralytics.com/datasets/detect/coco/

    Ultralytics. (n.d.). Integración con Roboflow. Ultralytics Docs. https://docs.ultralytics.com/es/integrations/roboflow/

    Ultralytics. (n.d.). Object detection. Ultralytics Docs. https://docs.ultralytics.com/tasks/detect/

    Ultralytics. (n.d.). Roboflow collect. Ultralytics Docs. https://docs.ultralytics.com/es/integrations/roboflow/#roboflow-collect

    Ultralytics. (n.d.). Train mode. Ultralytics Docs. https://docs.ultralytics.com/modes/train/

    Ultralytics. (n.d.). Ultralytics tutorial notebook. Google Colab. https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb#scrollTo=zR9ZbuQCH7FX

    Universidad Europea. (2025, December 30). Sports analytics: qué es y cómo optimiza el rendimiento. https://universidadeuropea.com/blog/sports-analytics/
