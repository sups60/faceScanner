import cv2
import face_recognition


# Function to capture and save the admin's face
def save_admin_face():
    video_capture = cv2.VideoCapture(0)
    admin_face = None

    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Admin Face Scan', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            admin_face = frame
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if admin_face is not None:
        cv2.imwrite('admin_face.jpg', admin_face)
        print("Admin face saved as admin_face.jpg")


# Function to scan and compare a client's face with the admin's face
def authorize_client():
    admin_image = face_recognition.load_image_file('admin_face.jpg')
    admin_face_encoding = face_recognition.face_encodings(admin_image)[0]

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)

        if len(face_locations) > 0:
            client_face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

            # Compare the client's face with the admin's face
            results = face_recognition.compare_faces([admin_face_encoding], client_face_encoding)

            if results[0]:
                print("Client authorized!")
                break

        cv2.imshow('Client Face Scan', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Press 's' to save the admin's face.")
    save_admin_face()
    print("Press 'q' to exit or wait for client face scan...")
    authorize_client()
