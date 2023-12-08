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
        cv2.line(overlay, (250, 360), (350, 360), (0, 0, 0), 2)
        cv2.line(overlay, (250, 390), (350, 390), (0, 0, 0), 2)
        cv2.line(overlay, (250, 420), (350, 420), (0, 0, 0), 2)
        cv2.line(overlay, (250, 450), (350, 450), (0, 0, 0), 2)
        cv2.line(overlay, (300, 390), (300, 420), (0, 0, 0), 2)
        cv2.line(overlay, (300, 420), (300, 450), (0, 0, 0), 2)
        cv2.line(overlay, (250, 180), (350, 180), (0, 0, 0), 2)
        cv2.line(overlay, (250, 210), (350, 210), (0, 0, 0), 2)
        cv2.line(overlay, (250, 240), (350, 240), (0, 0, 0), 2)
        cv2.line(overlay, (250, 270), (350, 270), (0, 0, 0), 2)
        cv2.line(overlay, (250, 300), (350, 300), (0, 0, 0), 2)
        cv2.line(overlay, (250, 330), (350, 330), (0, 0, 0), 2)
        cv2.line(overlay, (250, 180), (250, 450), (0, 0, 0), 2)
        cv2.line(overlay, (350, 180), (350, 450), (0, 0, 0), 2)

        frame1 = cv2.addWeighted(overlay, 0.4, frame1, 1, 0)


        # In case the system sees multiple hands this if statment deals with that and produces another hand overlay
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                # Below is Added Code to find and print to the shell the Location X-Y coordinates of Index Finger, Uncomment if desired
                # for point in handsModule.HandLandmark:

                # normalizedLandmark = handLandmarks.landmark[point]
                # pixelCoordinatesLandmark= drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 640, 480)

                # Using the Finger Joint Identification Image we know that point 8 represents the tip of the Index Finger
                # if point == 8:
                # print(point)
                # print(pixelCoordinatesLandmark)
                # print(normalizedLandmark)

        # Below shows the current frame to the desktop
        cv2.imshow("Frame", frame1);
        key = cv2.waitKey(1) & 0xFF

        # Below states that if the |q| is press on the keyboard it will stop the system
        if key == ord("q"):
            break