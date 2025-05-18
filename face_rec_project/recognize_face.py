import face_recognition
import pickle
import cv2
import os

# Load known face encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Path to the test image
image_path = "test.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ test.jpg not found in {os.getcwd()}")

# Load the test image and convert color to RGB
image = face_recognition.load_image_file(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect face locations and get face encodings
boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)

# Loop over detected faces
for encoding, box in zip(encodings, boxes):
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            matched_name = data["names"][i]
            counts[matched_name] = counts.get(matched_name, 0) + 1

        # Pick the name with the highest match count
        name = max(counts, key=counts.get)

    # Draw rectangle and label on the image
    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"✅ Detected: {name}")

# Convert back to BGR for OpenCV display
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Create a resizable window and set fixed size
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 800, 600)

# Show the image with boxes and names
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
