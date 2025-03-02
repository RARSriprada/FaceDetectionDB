import cv2
import numpy as np
import mysql.connector
import os

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1970',  # Update this with your correct password
    'database': 'facerecognitiondb'
}

# Function to save user to the database
def save_user_to_db(name, face_image):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Convert the face image to binary format
        _, buffer = cv2.imencode('.jpg', face_image)
        image_blob = buffer.tobytes()

        # Insert the name and image blob into the database
        cursor.execute("INSERT INTO Users (name, image_blob) VALUES (%s, %s)", (name, image_blob))
        conn.commit()
        print(f"User {name} has been saved to the database.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

# Function to capture reference images
def capture_reference_images():
    name = input("Enter your name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Position yourself in front of the camera. Press 'c' to capture your reference image. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray[y:y + h, x:x + w]

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == ord('c') and 'face_roi' in locals():
            save_user_to_db(name, face_roi)
            print("Image captured and saved.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to train recognizer
def train_recognizer():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT name, image_blob FROM Users")

        images = []
        labels = []
        label_map = {}
        current_label = 0

        for (name, image_blob) in cursor.fetchall():
            image_array = np.frombuffer(image_blob, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

            if name not in label_map:
                label_map[name] = current_label
                current_label += 1

            images.append(image)
            labels.append(label_map[name])

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images, np.array(labels))
        recognizer.save("face_recognizer.yml")
        print("Training completed and recognizer saved.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

# Function to recognize user
def recognize_user():
    if not os.path.exists("face_recognizer.yml"):
        print("No trained recognizer found. Train the recognizer first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer.yml")

    label_map = {}
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM Users")
        for i, (name,) in enumerate(cursor.fetchall()):
            label_map[i] = name
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Position yourself in front of the camera to recognize. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_roi)
            name = label_map.get(label, "Unknown")
            text = f"{name} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu
def main():
    while True:
        print("1. Capture reference images")
        print("2. Train recognizer")
        print("3. Recognize user")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            capture_reference_images()
        elif choice == '2':
            train_recognizer()
        elif choice == '3':
            recognize_user()
        else:
            print("Invalid choice. Exiting.")
            break

if __name__ == "__main__":
    main()
