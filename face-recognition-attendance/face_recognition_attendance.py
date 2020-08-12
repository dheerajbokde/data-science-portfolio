import face_recognition
import cv2
import numpy as np
import glob
import os
from datetime import datetime

# This is a demo of running face recognition on live video from your webcam.
#   1. Process each video frame at 1/4 resolution
#   2. Only detect faces in every other frame of video.
#   3. Mark attendance

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library.
def main():
    # Get a reference to webcam #0
    video_capture = cv2.VideoCapture(0)

    # Create arrays of known face encodings and their names
    known_face_names = []
    known_face_encodings = []
    for file_path in glob.glob('train_images/*.jpg'):
        image = face_recognition.load_image_file(file_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        face_name = os.path.splitext(os.path.basename(file_path))[0]
        known_face_names.append(face_name)
        known_face_encodings.append(face_encoding)

    process_this_frame = True

    while True:
        face_locations, face_names, frame = recognise_face_from_video(known_face_encodings, known_face_names, video_capture, process_this_frame)

        frame = display_video_result(face_locations, face_names, frame)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


# Mark attendance of the person with given name
def mark_attendance(name):
    file_path = 'test_result/attendance.csv'
    mode = 'r+' if os.path.exists(file_path) else 'w+'
    with open(file_path, mode) as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_str}')


# if match found display person name on the video frame
def display_video_result(face_locations, face_names, frame):
    # Display the results on video frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == 'Unknown':
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        else:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            # mark attendance
            mark_attendance(name)

        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    return frame


# (recognise face from video) find the match for the face in video frame
def recognise_face_from_video(known_face_encodings, known_face_names, video_capture, process_this_frame):
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    process_this_frame = not process_this_frame
    return face_locations, face_names, frame


if __name__ == "__main__":
    main()
