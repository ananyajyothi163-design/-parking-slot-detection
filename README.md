\# AI-Based Vehicle Parking Slot Detection and Monitoring System



\## 1. Project Overview

This project uses Artificial Intelligence and Computer Vision to detect and monitor empty and occupied parking spaces from images and camera input.



The system uses an RF-DETR Small object detection model to identify parking spaces and a Streamlit dashboard to display the results.



\## 2. Technologies Used

\- Python

\- RF-DETR Small

\- Computer Vision

\- OpenCV

\- Supervision

\- Streamlit

\- NumPy

\- Pillow



\## 3. Main Features

\- Upload parking images for detection.

\- Capture images using a camera.

\- Detect empty and occupied parking spaces.

\- Use tiled inference to improve detection coverage.

\- Display total, empty and occupied parking spaces.

\- Display annotated detection results.

\- Record parking location and monitoring details.

\- Download monitoring results as CSV.



\## 4. Model Information

\- Model: RF-DETR Small

\- Checkpoint: weights.pt

\- Confidence threshold: 0.30

\- IoU evaluation threshold: 0.50



\## 5. Dataset Evaluation Results

\- Test images processed: 1242

\- True Positives: 70524

\- False Positives: 300

\- False Negatives: 160

\- Precision: 99.58%

\- Recall: 99.77%

\- F1 Score: 99.67%



Note: These are IoU-based detection evaluation metrics, not overall classification accuracy.



\## 6. Class Mapping

\- Class ID 1: Empty Space

\- Class ID 2: Occupied Space



\## 7. Project Setup



Install the required packages:



```bash

pip install -r requirements.txt

```



Ensure that weights.pt is available in the project folder.



\## 8. Run the Application



```bash

streamlit run app.py

```



\## 9. Project Structure



```text

archive/

├── app.py

├── weights.pt

├── requirements.txt

├── README.md

├── train/

├── valid/

├── test/

└── venv/

```



\## 10. Future Enhancements

\- Live CCTV integration.

\- Real-time parking availability monitoring.

\- Drone-based parking surveillance.

\- Parking history and analytics.

