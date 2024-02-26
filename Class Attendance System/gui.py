# Import necessary packages
import os
import cv2
import time
import yagmail
import datetime
import pandas as pd
import PySimpleGUI as sg



# Function to verify the teacher
def teacher_verfication():
    sg.theme('Black')

    layout = [[sg.Text('Teacher Name:', size=(12, 1), font='Helvetica 14'),
               sg.InputText('', key='Teacher Name', font='Helvetica 14')],
              [sg.Text('Password:', size=(12, 1), font='Helvetica 14'),
               sg.InputText('', key='Password', password_char='*', font='Helvetica 14')],
              [sg.Button('Submit', button_color=('white', 'green'), font='Helvetica 14', size=(20, 1)),
               sg.Button('Cancel', button_color=('white', 'red'), font='Helvetica 14', size=(20, 1))]]

    window = sg.Window('Teacher Details', layout, element_justification='c')
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window.close()
            break
        elif event == 'Submit':
            name = values['Teacher Name']
            name = name.lower()
            pwd = values['Password']
            if os.path.isfile("attendance/teacher_details.csv"):
                teachers = pd.read_csv("attendance/teacher_details.csv")
                if not teachers[(teachers['Name'] == name) & (teachers['Password'] == pwd)].empty:
                    window.close()
                    return name
                else:
                    sg.popup('Cannot find the teacher')
            else:
                sg.popup('No teachers have been added yet')
    main_menu()


# Function to email the attendance
def email_attendance(teacher_name: str, sheet_path: str):
    teacher_details = pd.read_csv("attendance/teacher_details.csv")

    receiver = teacher_details.loc[teacher_details['Name'] == teacher_name]['Email Address'].values[0]
    body = "Attendance sheet is attached along with this email."  # email body

    # Sender information
    yag = yagmail.SMTP("nomanrasheed650@gmail.com", "pakistannavy106782")

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
def mark_attendance(teacher_name: str):
    start_time = time.time()

    sg.theme('Black')

    layout = [
        [sg.Image(key='-IMAGE-')],
    ]

    window = sg.Window('Mark Attendance', layout)

    # Load the face recognizer
    recognizer =cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("facial_recognition/model.yml")

    # Load the cascade
    face_detector = cv2.CascadeClassifier("facial_recognition/haarcascade_frontalface_default.xml")

    students = pd.read_csv("attendance/student_details.csv")
    col_names = ['CMS ID', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # To capture video from camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        sg.popup('There was an issue while opening the camera')

    cap.set(3, 640)  # set video width
    cap.set(4, 480)  # set video height

    # Min window size to be recognized as a face
    min_w = 0.1 * cap.get(3)
    min_h = 0.1 * cap.get(4)

    while cap.isOpened():
        event, values = window.read(timeout=0)

        if event == sg.WIN_CLOSED:
            break

        if time.time() - start_time >= 600:
            break

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

            conf_thresh = 45

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
        else:
            break

        # Update the image
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    hour, minute, second = time_stamp.split(":")
    file_name = "attendance/sheets/" + teacher_name + "_attendance_" + date + "_" + hour + "-" + minute + ".csv"
    attendance.to_csv(file_name, index=False)
    # email_attendance(teacher_name=teacher_name, sheet_path=file_name)
    window.close()


# Function to display main menu
def main_menu():
    sg.theme('Black')

    layout = [[sg.Text('Class Attendance System Using Facial Recognition', font='Helvetica 30',
                       justification='center')],
              [sg.Image('project.png', size=(650, 450))],
              [sg.Button("Mark Attendance", size=(82, 2), font='Helvetica 14', button_color=('white', '#303030'))]]

    main_window = sg.Window('Class Attendance System Using Facial Recognition', layout, auto_size_buttons=True,
                            element_justification='c')

    while True:
        event, values = main_window.read(timeout=0)
        if event == "Quit" or event == "Exit" or event == sg.WIN_CLOSED:
            main_window.close()
            break
        elif event == "Mark Attendance":
            main_window.close()
        mark_attendance("ali")



if __name__ == "__main__":
    main_menu()
