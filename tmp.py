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
        # In case the system sees multiple hands this if statment deals with that and produces another hand overlay
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                for point in handsModule.HandLandmark:
                    normalizedLandmark = handLandmarks.landmark[point]
                    #pixelCoordinatesLandmark= drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 1000, 1000)
                    if point == 4:
                        right_x = int(normalizedLandmark.x * 1000)
                        right_y = int(normalizedLandmark.y * 1000)
                        if right_y > 650:
                            print(point)
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





                for hand_landmarks in results.multi_hand_landmarks:
                    finger1_x = int(hand_landmarks.landmark[4].x * 1000)
                    finger2_x = int(hand_landmarks.landmark[8].x * 1000)
                    finger3_x = int(hand_landmarks.landmark[12].x * 1000)
                    finger4_x = int(hand_landmarks.landmark[16].x * 1000)
                    finger5_x = int(hand_landmarks.landmark[20].x * 1000)

                    finger1_y = int(hand_landmarks.landmark[4].y * 1000)
                    finger2_y = int(hand_landmarks.landmark[8].y * 1000)
                    finger3_y = int(hand_landmarks.landmark[12].y * 1000)
                    finger4_y = int(hand_landmarks.landmark[16].y * 1000)
                    finger5_y = int(hand_landmarks.landmark[20].y * 1000)








        # Below shows the current frame to the desktop
        cv2.imshow("Frame", frame1);
        key = cv2.waitKey(1) & 0xFF

        # Below states that if the |q| is press on the keyboard it will stop the system
        if key == ord("q"):
            break