import face_recognition
from PIL import Image, ImageDraw
import numpy as np

barakat_image = face_recognition.load_image_file("./known/barakat.jpg")
barakat_face_encoding = face_recognition.face_encodings(barakat_image)[0]

treka_image = face_recognition.load_image_file("./known/treka.jpg")
treka_face_encoding = face_recognition.face_encodings(treka_image)[0]

z3ama_image = face_recognition.load_image_file("./known/z3ama.jpg")
z3ama_face_encoding = face_recognition.face_encodings(z3ama_image)[0]

known_face_encodings = [barakat_face_encoding,
                        treka_face_encoding, z3ama_face_encoding]

known_faces_names = ["Mohamed Barkat ",
                     "Mohamed abu treka", "el z3ama Adel Shakl"]


image = input("please enter image number : ")
unknown_image = face_recognition.load_image_file(f'./un_known/{image}.jpg')

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding)
    name = "unknown"

    face_distance = face_recognition.face_distance(
        known_face_encodings, face_encoding)

    best_match_index = np.argmin(face_distance)

    if matches[best_match_index]:
        name = known_faces_names[best_match_index]

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10),
                   (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5),
              name, fill=(255, 255, 255, 255))
del draw

pil_image.show()
