import cv2
import time

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the reference image (if any)
ref_image_path = "reference.jpg"
try:
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise Exception("Unable to load reference image")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the time limit
time_limit = 60 * 60  # 60 minutes

start_time = time.time()

while True:
    # Check if the time limit has been exceeded
    if time.time() - start_time > time_limit:
        break

    # Capture the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no face is detected
    if len(faces) == 0:
        print("No face detected!")

    # If multiple faces are detected
    elif len(faces) > 1:
        print("Multiple faces detected!")

    # If a single face is detected
    else:
        # Check if the face matches the reference image
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            if ref_image is not None:
                # Convert the reference image to grayscale
                ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
                
                # Resize the reference image to match the face size
                ref_gray = cv2.resize(ref_gray, (w, h))

                # Compare the faces
                sim = cv2.matchTemplate(roi_gray, ref_gray, cv2.TM_CCORR_NORMED)
                threshold = 0.5
                if sim > threshold:
                    print("Same face detected!")
                else:
                    print("Different face detected!")
                    cv2.imwrite('captured_image.jpg', frame)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()