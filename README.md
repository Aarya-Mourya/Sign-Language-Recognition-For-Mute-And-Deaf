# Sign-Language-Recognition-For-Mute-And-Deaf

<img src="https://mediapipe.dev/images/mobile/hand_landmarks.png" alt="Hand Landmarks">

An American Sign Language Recognition System has been developed to accurately interpret the 26 alphabets, benefiting individuals who are mute or deaf. Leveraging Python, OpenCV, and Google's Mediapipe library for hand landmark detection, precise hand movements are captured. Matplotlib facilitates data visualization, while NumPy handles numerical computations. Through the integration of a Random Forest Classifier from Scikit-learn, a classification accuracy of 100% is achieved, ensuring correct identification of all samples. This project exemplifies the fusion of advanced machine learning techniques and computer vision to create a powerful communication tool for the mute and deaf community.


## Features

#### 1. Real-time Hand Detection and Tracking
- **Use of Mediapipe**: Employs Google’s Mediapipe library to detect and track hand landmarks in real-time, providing accurate and efficient hand gesture recognition.
- **OpenCV Integration**: Utilizes OpenCV for capturing video feed from the webcam, ensuring seamless integration with the Mediapipe hand detection module.

#### 2. High-Accuracy Classification
- **Random Forest Classifier**: Implemented using Scikit-learn, the classifier was trained to recognize 26 ASL alphabets with 100% accuracy on the test samples.
- **Feature Extraction**: Hand landmarks captured by Mediapipe are processed to extract relevant features for accurate classification.

#### 3. User-Friendly Interface
- **Live Feedback**: Displays the recognized alphabet on the screen in real-time, providing immediate feedback to the user.
- **Visualization Tools**: Utilizes Matplotlib to visualize hand landmarks and recognition results, aiding in better understanding and debugging of the system.

#### 4. Data Handling and Processing
- **NumPy for Computations**: Leverages NumPy for efficient numerical computations and data handling.
- **Data Augmentation**: Implements data augmentation techniques to increase the diversity of the training dataset, improving the robustness of the classifier.

#### 5. Robustness and Scalability
- **Cross-Platform Compatibility**: Developed to run on multiple operating systems including Windows, macOS, and Linux.
- **Modular Code Structure**: Designed with a modular approach, allowing easy updates and integration of additional features or improvements.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the required libraries using pip:

```bash
pip install opencv-python mediapipe scikit-learn matplotlib numpy
