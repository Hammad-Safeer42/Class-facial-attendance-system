# Import necessary packages
import os
import cv2
import time
import shutil
import yagmail
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from unicodedata import numeric


# Function to display program's title
def title():
    # Clear the screen
    os.system('cls')
    # Title of the program
    print("***********************************")
    print("***** Class Attendance System *****")
    print("***********************************")


# Function to display main menu
def main_menu():
    title()
    print(10 * "-", "MAIN MENU", 10 * "-")
    print("[1] Check Camera")
    print("[2] Mark Attendance")
    print("[3] Manage Students")
    print("[4] Manage Teachers")
    print("[5] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                check_camera()
                break
            elif choice == 2:
                mark_attendance()
                break
            elif choice == 3:
                manage_students()
                break
            elif choice == 4:
                manage_teachers()
                break
            elif choice == 5:
                print("Thank you")
                break
            else:
                print("Invalid Choice. Try Again")
                main_menu()
        except ValueError:
            print("Invalid Choice. Try Again")
    exit()


# Function to check camera
def check_camera():
    # Load the cascade
    face_detector = cv2.CascadeClassifier('facial_recognition/haarcascade_frontalface_default.xml')

    # To capture video from camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("There was an issue while opening the camera")

    while cap.isOpened():
        # Read the frame
        ret, img = cap.read()
        if ret:
            # Flips the original frame about y-axis
            img = cv2.flip(img, 1)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)

            # Display
            cv2.imshow('Camera Check', img)

            # Stop if escape key or 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release the video capture object & destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    key = input("Enter any key to return to main menu ")
    main_menu()


# Function to check if input is a valid number
def is_number(string: str):
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        numeric(string)
        return True
    except (TypeError, ValueError):
        pass


# Function to return training images and labels
def imgs_and_labels(path: str):
    # Create empty list for faces
    faces = []
    # Create empty list for CMS IDs
    cms_ids = []
    # Obtain a list of directories & files available inside the path
    _, directories, _ = next(os.walk(path))
    for directory in directories:
        # Obtain a list of files available within the subdirectory
        _, _, files = next(os.walk(path + '/' + directory))
        # Loop through each file within the subdirectory
        for file in files:
            # Load the image and convert it to gray scale
            pill_img = Image.open(path + '/' + directory + '/' + file).convert('L')
            # Convert the PIL image into numpy array
            np_img = np.array(pill_img, 'uint8')
            # Get the CMS ID
            cms_id = int(directory.split("_")[-1])
            # Append the face to faces list
            faces.append(np_img)
            # Append the cms_id to CMS ids list
            cms_ids.append(cms_id)
    return faces, cms_ids


# Function to view all students
def view_students():
    if os.path.isfile("attendance/student_details.csv"):
        student_details = pd.read_csv("attendance/student_details.csv")
        if not student_details.empty:
            print(student_details)
        else:
            print("No students have been added yet.")
    else:
        print("No students have been added yet.")


# Function to add a student
def add_student():
    cms_id = input("CMS ID: ")
    while not is_number(cms_id):
        print("Please enter valid CMS ID")
        cms_id = input("CMS ID: ")

    name = input("Name: ")

    print("Capturing the face...")
    # Make a folder for the student if it doesnot exist
    student = name.replace(" ", "_") + "_" + cms_id
    if not os.path.isdir("facial_recognition/faces/" + student):
        os.mkdir("facial_recognition/faces/" + student)

    # Load the cascade
    face_detector = cv2.CascadeClassifier("facial_recognition/haarcascade_frontalface_default.xml")

    # To capture video from camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("There was an issue while opening the camera")

    sample_num = 0

    while cap.isOpened():
        # Read the frame
        ret, img = cap.read()
        if ret:
            # Flips the original frame about y-axis
            img = cv2.flip(img, 1)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                # Increment the sample number
                sample_num += 1
                # Draw the rectangle around the face
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                # Save the captured face
                cv2.imwrite("facial_recognition/faces/" + student + "/" +
                            str(sample_num) + ".jpg", gray[y:y + h, x:x + w])
                # Display the frame
                cv2.imshow(student, img)

            # Stop after 100 milliseconds or if 'q' is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # Exit if the number of samples is greater or equal to 100
            elif sample_num >= 500:
                break
        else:
            break
    # Release the video capture object & destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the student details
    if os.path.isfile("attendance/student_details.csv"):
        student_details = pd.read_csv("attendance/student_details.csv")
        student_details = pd.concat([student_details, pd.DataFrame({'CMS ID': [cms_id], 'Name': [name]})],
                                    ignore_index=True,
                                    axis=0)
        student_details.drop_duplicates(subset=['CMS ID'], inplace=True)
        student_details.to_csv('attendance/student_details.csv', index=False)
    else:
        student_details = pd.DataFrame(data={'CMS ID': [cms_id], 'Name': [name]})
        student_details.to_csv('attendance/student_details.csv', index=False)

    print("Training the model")
    # Train on the images & save the model
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, cms_ids = imgs_and_labels("facial_recognition/faces")
    recognizer.train(faces, np.array(cms_ids))
    recognizer.save("facial_recognition/model.yml")

    print("Student Added!")
    key = input("Enter any key to return ")
    manage_students()


# Function to remove a student
def remove_student():
    print("Enter CMS ID of the student to be removed")
    cms_id = input("CMS ID: ")
    while not is_number(cms_id):
        print("Please enter valid CMS ID \n")
        cms_id = input("CMS ID: ")
    # Read the student details
    students = pd.read_csv("attendance/student_details.csv")
    # If the student exists
    if not students[students['CMS ID'] == int(cms_id)].empty:
        # Get the student's name
        name = students.loc[students['CMS ID'] == int(cms_id)]['Name'].values[0]

        # Remove student's data from student_details.csv
        students.drop(index=students[students["CMS ID"] == int(cms_id)].index, inplace=True)
        students.to_csv("attendance/student_details.csv", index=False)

        # Remove student's pictures
        try:
            shutil.rmtree("facial_recognition/faces/" + name.replace(" ", "_") + "_" + cms_id)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        # Retrain the model
        faces, cms_ids = imgs_and_labels("facial_recognition/faces")
        if len(faces) != 0 and len(cms_ids) != 0:
            recognizer = cv2.face_LBPHFaceRecognizer.create()
            recognizer.train(faces, np.array(cms_ids))
            recognizer.save("facial_recognition/model.yml")

        print("Student Removed!")
    else:
        print("Student not found")

    key = input("Enter any key to return ")
    manage_students()


# Function to manage students
def manage_students():
    print(10 * "-", "MANAGE STUDENTS", 10 * "-")
    print("[1] View all Students")
    print("[2] Add a Student")
    print("[3] Remove a Student")
    print("[4] Go to Main Menu")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                view_students()
                manage_students()
                break
            elif choice == 2:
                add_student()
                break
            elif choice == 3:
                view_students()
                remove_student()
                break
            elif choice == 4:
                main_menu()
                break
            else:
                print("Invalid Choice. Try Again!")
                main_menu()
        except ValueError:
            print("Invalid Choice. Try Again!")
    exit()


# Function to view all teachers
def view_teachers():
    if os.path.isfile("attendance/teacher_details.csv"):
        teacher_details = pd.read_csv("attendance/teacher_details.csv")
        if not teacher_details.empty:
            print(teacher_details)
        else:
            print("No teachers have been added yet.")
    else:
        print("No teachers have been added yet.")


# Function to add a teacher
def add_teacher():
    name = input("Name: ")
    name = name.lower()
    email = input("Email Address: ")
    pwd = input("Set a password: ")

    # Save the teacher details
    if os.path.isfile("attendance/teacher_details.csv"):
        teacher_details = pd.read_csv("attendance/teacher_details.csv")
        teacher_details = pd.concat([teacher_details, pd.DataFrame({'Name': [name], 'Email Address': [email],
                                                                    'Password': [pwd]})],
                                    ignore_index=True,
                                    axis=0)
        teacher_details.drop_duplicates(subset=['Name'], inplace=True)
        teacher_details.to_csv('attendance/teacher_details.csv', index=False)
    else:
        teacher_details = pd.DataFrame(data={'Name': [name], 'Email Address': [email], 'Password': [pwd]})
        teacher_details.to_csv('attendance/teacher_details.csv', index=False)

    print("Teacher Added!")
    key = input("Enter any key to return ")
    manage_teachers()


# Function to remove a teacher
def remove_teacher():
    print("Enter the name of the teacher to be removed")
    name = input("Name: ")
    name = name.lower()

    # Read the student details
    teachers = pd.read_csv("attendance/teacher_details.csv")
    # If the student exists
    if not teachers[teachers['Name'] == name].empty:
        # Remove student's data from student_details.csv
        teachers.drop(index=teachers[teachers["Name"] == name].index, inplace=True)
        teachers.to_csv("attendance/teacher_details.csv", index=False)

        print("Teacher Removed!")
    else:
        print("Teacher not found")

    key = input("Enter any key to return ")
    manage_teachers()


# Function to manage teachers
def manage_teachers():
    print(10 * "-", "MANAGE TEACHERS", 10 * "-")
    print("[1] View all Teachers")
    print("[2] Add a Teacher")
    print("[3] Remove a Teacher")
    print("[4] Go to Main Menu")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                view_teachers()
                manage_teachers()
                break
            elif choice == 2:
                add_teacher()
                break
            elif choice == 3:
                view_teachers()
                remove_teacher()
                break
            elif choice == 4:
                main_menu()
                break
            else:
                print("Invalid Choice. Try Again!")
                main_menu()
        except ValueError:
            print("Invalid Choice. Try Again!")
    exit()


# Function to verify the teacher
def teacher_verfication(name: str, pwd: str):
    if os.path.isfile("attendance/teacher_details.csv"):
      teacher_details = pd.read_csv("attendance/teacher_details.csv")
    if not teacher_details[(teacher_details['Name'] == name) & (teacher_details['Password'] == pwd)].empty:
           return True
    else:
     print("No teachers have been added yet.")
   # else:
   # print("No teachers have been added yet.")


# Function to email the attendance
def email_attendance(teacher_name: str, sheet_path: str):
    teacher_details = pd.read_csv("attendance/teacher_details.csv")

    receiver = teacher_details.loc[teacher_details['Name'] == teacher_name]['Email Address'].values[0]
    body = "Attendance sheet is attached along with this email."  # email body

    # Sender information
    yag = yagmail.SMTP("", "")

    # Send the email to teacher
    yag.send(
        to=receiver,
        subject=teacher_name + " - Attendance Report",  # email subject
        contents=body,  # email body
        attachments=sheet_path,  # file attached
    )

    # Send the email to HOUGP EPE
    yag.send(
        to="ayahya@pnec.nust.edu.pk",
        subject=teacher_name + " - Attendance Report",  # email subject
        contents=body,  # email body
        attachments=sheet_path,  # file attached
    )


# Function to mark attendance
def mark_attendance():
    teacher_name = input("Teacher Name: ")
    teacher_name = teacher_name.lower()
    pwd = input("Password: ")

    if teacher_verfication(name=teacher_name, pwd=pwd):
        # Load the face recognizer
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        recognizer.read("facial_recognition/model.yml")

        # Load the cascade
        face_detector = cv2.CascadeClassifier("facial_recognition/haarcascade_frontalface_default.xml")

        students = pd.read_csv("attendance/student_details.csv")
        col_names = ['CMS ID', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # To capture video from camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 640)  # set video width
        cap.set(4, 480)  # set video height

        # Min window size to be recognized as a face
        min_w = 0.1 * cap.get(3)
        min_h = 0.1 * cap.get(4)

        if not cap.isOpened():
            print("There was an issue while opening the camera")

        while cap.isOpened():
            # Read the frame
            ret, img = cap.read()
            if ret:
                # Flips the original frame about y-axis
                img = cv2.flip(img, 1)

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect the faces
                faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(int(min_w), int(min_h)),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
                conf_thresh = 65

                for (x, y, w, h) in faces:
                    # Draw the rectangle around the face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                    cms_id, conf = recognizer.predict(gray[y:y + h, x:x + w])

                    if conf < 100:
                        name = students.loc[students['CMS ID'] == cms_id]['Name'].values
                        confstr = "  {0}%".format(round(100 - conf))
                        student = str(cms_id) + "-" + name

                    else:
                        cms_id = '  Unknown  '
                        student = str(cms_id)
                        confstr = "  {0}%".format(round(100 - conf))

                    if (100 - conf) > conf_thresh:
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        name = str(name)[2:-2]
                        attendance.loc[len(attendance)] = [cms_id, name, date, time_stamp]

                    student = str(student)[2:-2]
                    if (100 - conf) > conf_thresh:
                        cv2.putText(img, str(student), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

                    if (100 - conf) > conf_thresh:
                        cv2.putText(img, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
                    elif (100 - conf) > 50:
                        cv2.putText(img, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
                    else:
                        cv2.putText(img, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

                attendance = attendance.drop_duplicates(subset=['CMS ID'], keep='first')
                cv2.imshow('Attendance', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            else:
                break

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        hour, minute, second = time_stamp.split(":")
        file_name = "attendance/sheets/" + teacher_name + "_attendance_" + date + "_" + hour + "-" + minute + ".csv"
        attendance.to_csv(file_name, index=False)
        cap.release()
        cv2.destroyAllWindows()

        # Email the attendance
        # email_attendance(teacher_name=teacher_name, sheet_path=file_name)

        print("Attendance Successful!")

    key = input("Enter any key to return to main menu ")
    main_menu()


if __name__ == "__main__":
    main_menu()
