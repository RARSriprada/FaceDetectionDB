# Face Detection and Recognition System

This project is a Face Detection and Recognition system using OpenCV and MySQL. It captures, trains, and recognizes users based on facial features.

## Features
- Detects faces using OpenCV's Haar cascade classifier.
- Recognizes users based on pre-trained data.
- Stores user information in a MySQL database.
- Provides options to capture reference images, train the recognizer, and recognize users.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MySQL Connector (`mysql-connector-python`)
- NumPy (`numpy`)

## Installation
1. Clone this repository:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd FaceDetectionDB
   ```
3. Install dependencies:
   ```sh
   pip install opencv-python mysql-connector-python numpy
   ```
4. Set up MySQL database and update the `db_config` variable in the script with your credentials.


1. Run the script:
   ```sh
   python FacedetectionDB.py
   ```
2. Choose an option from the menu:
   - `1` to capture reference images.
   - `2` to train the recognizer.
   - `3` to recognize a user.

3. Position yourself in front of the camera for recognition. Press `q` to exit.

NOTE: Update The passsword Of your SQL in the FaceDetectionDB.py file

