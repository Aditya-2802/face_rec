import os
import face_recognition
import pickle

dataset_path = "dataset"
encoding_file = "encodings.pickle"

known_encodings = []
known_names = []

# Loop through each person
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save encodings to a file
data = {"encodings": known_encodings, "names": known_names}
with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print("Encoding complete. Data saved to", encoding_file)
