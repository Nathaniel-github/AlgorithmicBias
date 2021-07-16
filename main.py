import face_recognition
import pickle
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("labels_utf8.csv")
encodings = []
file_name = "encodings.pkl"
for name in tqdm(df['ID']):
    encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(f"mugshots/front/front/{name}"))[0])

print("Images encoded")
print(f"Total encodings: {len(encodings)}")

with open(file_name, 'wb+') as f:
    pickle.dump(encodings, f)
    print(f"File saved as: {file_name}")
