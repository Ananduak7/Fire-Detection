import cv2
from ultralytics import YOLO
import cvzone
import math
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Function to perform fire detection
def fire_detection_loop():
    global cap, model, classnames, root, video_panel
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))  # Resize frame to fit in UI

        # Perform object detection
        result = model(frame, stream=True)

        # Getting bbox, confidence, and class names information
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)

        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format for tkinter
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        video_panel.configure(image=frame)
        video_panel.image = frame

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Function to start the webcam feed and fire detection
def start_detection():
    # Load YOLO model trained on fire detection
    global model, cap, classnames
    model = YOLO('fire.pt')

    # Running real-time detection from webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify a different camera index if needed

    # Reading the classes
    classnames = ['fire']

    # Create a thread for fire detection
    thread = threading.Thread(target=fire_detection_loop)
    thread.start()

# Function to stop the webcam feed and fire detection
def stop_detection():
    cap.release()
    root.quit()

# Create a Tkinter window
root = tk.Tk()
root.title("Fire Detection")
root.configure(bg="red")

# Create a panel to display the video feed
video_panel = tk.Label(root)
video_panel.pack(padx=10, pady=10)

# Create a button to start detection
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

# Create a button to stop detection
stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
