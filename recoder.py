# Import the necessary Packages for this software to run
import mediapipe
import cv2

# Use MediaPipe to draw the hand framework over the top of hands it identifies in Real-Time
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# Use CV2 Functionality to create a Video stream and add some values
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=2) as hands:
    # Create an infinite loop which will produce the live feed to our desktop and that will search for hands
    while cap.isOpened():
        ret, frame = cap.read()
        flipped = cv2.flip(frame, flipCode = 3)
        frame1 = cv2.resize(flipped, (1000, 1000))
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        overlay = frame1.copy()
        cv2.line(overlay, (0, 650), (1000, 650), (0, 0, 0), 2)

        gesture_text = 'Cant found hand'
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                finger1_right_y = 0
                finger1_right_x = 0
                finger2_right_y = 0
                finger2_right_x = 0
                finger3_right_y = 0
                finger3_right_x = 0
                finger4_right_y = 0
                finger4_right_x = 0
                finger5_right_y = 0
                finger5_right_x = 0

                finger1_left_x = 0
                finger1_left_y = 0
                finger2_left_x = 0
                finger2_left_y = 0
                finger3_left_x = 0
                finger3_left_y = 0
                finger4_left_x = 0
                finger4_left_y = 0
                finger5_left_x = 0
                finger5_left_y = 0
                for point in handsModule.HandLandmark:
                    normalizedLandmark = handLandmarks.landmark[point]
                    #pixelCoordinatesLandmark= drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 1000, 1000)
                    if point == 4:
                        right_x = int(normalizedLandmark.x * 1000)
                        right_y = int(normalizedLandmark.y * 1000)
                        if right_y > 650:
                            # print(point)
                            finger1_right_y = right_y
                            finger1_right_x = right_x
                            # print(pixelCoordinatesLandmark)
                            # print(normalizedLandmark)
                            right_x -= 30
                            cv2.rectangle(overlay, (right_x - 30, right_y - 175), (right_x + 30, right_y + 225), (255,255,255), -1)
                            cv2.rectangle(overlay, (right_x - 60, right_y - 225), (right_x, right_y - 175), (255, 255, 255), -1)
                            cv2.line(overlay, (right_x - 60, right_y - 225), (right_x, right_y - 225), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 60, right_y - 225), (right_x - 60, right_y - 175), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x, right_y - 225), (right_x, right_y - 175), (0, 0, 0),2)
                            cv2.line(overlay, (right_x - 60, right_y - 175), (right_x - 30, right_y - 175), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y - 175), (right_x + 30, right_y - 175), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y - 125), (right_x + 30, right_y - 125), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y - 75), (right_x + 30, right_y - 75), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y - 25), (right_x + 30, right_y - 25), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y + 25), (right_x + 30, right_y + 25), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y + 75), (right_x + 30, right_y + 75), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y + 125), (right_x + 30, right_y + 125), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y + 175), (right_x + 30, right_y + 175), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y + 225), (right_x + 30, right_y + 225), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x, right_y + 125), (right_x, right_y + 175), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x, right_y + 175), (right_x, right_y + 225), (0, 0, 0), 2)
                            cv2.line(overlay, (right_x - 30, right_y - 175), (right_x - 30, right_y + 225), (0, 0, 0),2)
                            cv2.line(overlay, (right_x + 30, right_y - 175), (right_x + 30, right_y + 225), (0, 0, 0),2)
                            frame1 = cv2.addWeighted(overlay, 0.4, frame1, 1, 0)
                        else:
                            finger1_right_y = right_y
                            finger1_right_x = right_x

                    if point == 8:
                        if int(normalizedLandmark.y * 1000) > finger1_right_y:
                            finger2_right_x = int(normalizedLandmark.x * 1000)
                            finger2_right_y = int(normalizedLandmark.y * 1000)
                        else:
                            finger2_left_x = int(normalizedLandmark.x * 1000)
                            finger2_left_y = int(normalizedLandmark.y * 1000)

                    if point == 12:
                        if int(normalizedLandmark.y * 1000) > finger1_right_y:
                            finger3_right_x = int(normalizedLandmark.x * 1000)
                            finger3_right_y = int(normalizedLandmark.y * 1000)
                        else:
                            finger3_left_x = int(normalizedLandmark.x * 1000)
                            finger3_left_y = int(normalizedLandmark.y * 1000)
                    if point == 16:
                        if int(normalizedLandmark.y * 1000) > finger1_right_y:
                            finger4_right_x = int(normalizedLandmark.x * 1000)
                            finger4_right_y = int(normalizedLandmark.y * 1000)
                        else:
                            finger4_left_x = int(normalizedLandmark.x * 1000)
                            finger4_left_y = int(normalizedLandmark.y * 1000)
                    if point == 20:
                        if int(normalizedLandmark.y * 1000) > finger1_right_y:
                            finger5_right_x = int(normalizedLandmark.x * 1000)
                            finger5_right_y = int(normalizedLandmark.y * 1000)
                        else:
                            finger5_left_x = int(normalizedLandmark.x * 1000)
                            finger5_left_y = int(normalizedLandmark.y * 1000)

                    print("1",finger1_right_y, finger1_right_x)
                    print("2",finger5_right_y, finger5_right_x)
                    print("", finger5_left_y, finger5_left_x)

                    if finger1_right_y + 175 < finger5_right_y < finger1_right_y + 255:
                        if finger1_right_x - 30 < finger5_right_x < finger1_right_x + 30:
                            gesture_text = 'do'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 225), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 225), (0,0,255), 3)
                    elif finger1_right_y + 125 < finger4_right_y < finger1_right_y + 175:
                        if finger1_right_x - 30 < finger4_right_x < finger1_right_x + 30:
                            gesture_text = 're'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x , finger1_right_y + 175), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 175), (0, 0, 255), 3)
                    elif finger1_right_y + 75 < finger3_right_y < finger1_right_y + 125:
                        if finger1_right_x - 30 < finger3_right_x < finger1_right_x + 30:
                            gesture_text = 'mi'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 125), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 125), (0, 0, 255), 3)
                    elif finger1_right_y + 25 < finger2_right_y < finger1_right_y + 75:
                        if finger1_right_x - 30 < finger2_right_x < finger1_right_x + 30:
                            gesture_text = 'pa'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 75), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),(finger1_right_x, finger1_right_y + 75), (0, 0, 255), 3)

                    elif finger1_right_y - 25 < finger5_left_y < finger1_right_y + 25:
                        if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                            gesture_text = 'su'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),(finger1_right_x, finger1_right_y + 25), (0, 0, 255), 3)
                    elif finger1_right_y - 50 < finger4_left_y < finger1_right_y - 25:
                        if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                            gesture_text = 'ra'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),(finger1_right_x, finger1_right_y - 25), (0, 0, 255), 3)
                    elif finger1_right_y - 75 < finger5_left_y < finger1_right_y - 50:
                        if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                            gesture_text = 'si'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),(finger1_right_x, finger1_right_y - 50), (0, 0, 255), 3)
                    elif finger1_right_y - 125 < finger5_left_y < finger1_right_y - 75:
                        if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                            gesture_text = 'do2'
                            # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                            cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),(finger1_right_x, finger1_right_y - 75), (0, 0, 255), 3)




        cv2.putText(frame1, text='dd: {}'.format(gesture_text)
                    , org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)

        # Below shows the current frame to the desktop
        cv2.imshow("Frame", frame1);
        key = cv2.waitKey(1) & 0xFF

        # Below states that if the |q| is press on the keyboard it will stop the system
        if key == ord("q"):
            break