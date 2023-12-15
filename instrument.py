# Import the necessary Packages for this software to run
import mediapipe
import cv2
import pyaudio
import numpy as np
import pygame
import os

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
pygame.mixer.init()

CHUNK = 2 ** 10
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=2) as hands:
    control = 'p'
    gesture_text = 'Piano'
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
    while cap.isOpened():
        data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
        vol = int(np.average(np.abs(data)))
        print(vol)

        ret, frame = cap.read()
        flipped = cv2.flip(frame, flipCode = 3)
        frame1 = cv2.resize(flipped, (1000, 1000))
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        overlay = frame1.copy()
        if control == 'r':
            gesture_text = 'Recoder'
            cv2.line(overlay, (0, 650), (1000, 650), (0, 0, 0), 2)
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                    for point in handsModule.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        # pixelCoordinatesLandmark= drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 1000, 1000)
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
                                cv2.rectangle(overlay, (right_x - 30, right_y - 175), (right_x + 30, right_y + 225),
                                              (255, 255, 255), -1)
                                cv2.rectangle(overlay, (right_x - 60, right_y - 225), (right_x, right_y - 175),
                                              (255, 255, 255), -1)
                                cv2.line(overlay, (right_x - 60, right_y - 225), (right_x, right_y - 225), (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 60, right_y - 225), (right_x - 60, right_y - 175),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x, right_y - 225), (right_x, right_y - 175), (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 60, right_y - 175), (right_x - 30, right_y - 175),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 30, right_y - 175), (right_x + 30, right_y - 175),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 30, right_y - 125), (right_x + 30, right_y - 125),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 30, right_y - 75), (right_x + 30, right_y - 75), (0, 0, 0),
                                         2)
                                cv2.line(overlay, (right_x - 30, right_y - 25), (right_x + 30, right_y - 25), (0, 0, 0),
                                         2)
                                cv2.line(overlay, (right_x - 30, right_y + 25), (right_x + 30, right_y + 25), (0, 0, 0),
                                         2)
                                cv2.line(overlay, (right_x - 30, right_y + 75), (right_x + 30, right_y + 75), (0, 0, 0),
                                         2)
                                cv2.line(overlay, (right_x - 30, right_y + 125), (right_x + 30, right_y + 125),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 30, right_y + 175), (right_x + 30, right_y + 175),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 30, right_y + 225), (right_x + 30, right_y + 225),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x, right_y + 125), (right_x, right_y + 175), (0, 0, 0), 2)
                                cv2.line(overlay, (right_x, right_y + 175), (right_x, right_y + 225), (0, 0, 0), 2)
                                cv2.line(overlay, (right_x - 30, right_y - 175), (right_x - 30, right_y + 225),
                                         (0, 0, 0), 2)
                                cv2.line(overlay, (right_x + 30, right_y - 175), (right_x + 30, right_y + 225),
                                         (0, 0, 0), 2)
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

                        if vol > 90:
                            if finger1_right_y + 175 < finger5_right_y < finger1_right_y + 255:
                                if finger1_right_x - 30 < finger5_right_x < finger1_right_x + 30:
                                    gesture_text = 'do'
                                    sound_dir = r'.\recorder\do.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 225), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 225), (0, 0, 255), 3)
                            elif finger1_right_y + 125 < finger4_right_y < finger1_right_y + 175:
                                if finger1_right_x - 30 < finger4_right_x < finger1_right_x + 30:
                                    gesture_text = 're'
                                    sound_dir = r'.\recorder\re.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x , finger1_right_y + 175), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 175), (0, 0, 255), 3)
                            elif finger1_right_y + 75 < finger3_right_y < finger1_right_y + 125:
                                if finger1_right_x - 30 < finger3_right_x < finger1_right_x + 30:
                                    gesture_text = 'mi'
                                    sound_dir = r'.\recorder\mi.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 125), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 125), (0, 0, 255), 3)
                            elif finger1_right_y + 25 < finger2_right_y < finger1_right_y + 75:
                                if finger1_right_x - 30 < finger2_right_x < finger1_right_x + 30:
                                    gesture_text = 'pa'
                                    sound_dir = r'.\recorder\fa.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 75), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 75), (0, 0, 255), 3)

                            elif finger1_right_y - 25 < finger5_left_y < finger1_right_y + 25:
                                if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                                    gesture_text = 'sol'
                                    sound_dir = r'.\recorder\sol.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 25), (0, 0, 255), 3)
                            elif finger1_right_y - 50 < finger4_left_y < finger1_right_y - 25:
                                if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                                    gesture_text = 'la'
                                    sound_dir = r'.\recorder\la.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y - 25), (0, 0, 255), 3)
                            elif finger1_right_y - 75 < finger5_left_y < finger1_right_y - 50:
                                if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                                    gesture_text = 'ti'
                                    sound_dir = r'.\recorder\si.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y - 50), (0, 0, 255), 3)
                            elif finger1_right_y - 125 < finger5_left_y < finger1_right_y - 75:
                                if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                                    gesture_text = 'do2'
                                    sound_dir = r'.\recorder\highdo.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y - 75), (0, 0, 255), 3)

        if control == 'p':
            gesture_text = 'Piano'
            cv2.rectangle(overlay, (220, 490), (780, 810), (255, 255, 255), -1)
            cv2.line(overlay, (220, 490), (780, 490), (0, 0, 0), 2)
            cv2.line(overlay, (220, 490), (220, 810), (0, 0, 0), 2)
            cv2.line(overlay, (220, 810), (780, 810), (0, 0, 0), 2)
            cv2.line(overlay, (780, 490), (780, 810), (0, 0, 0), 2)
            cv2.line(overlay, (300, 650), (300, 810), (0, 0, 0), 2)
            cv2.line(overlay, (380, 650), (380, 810), (0, 0, 0), 2)
            cv2.line(overlay, (460, 650), (460, 810), (0, 0, 0), 2)
            cv2.line(overlay, (540, 650), (540, 810), (0, 0, 0), 2)
            cv2.line(overlay, (620, 650), (620, 810), (0, 0, 0), 2)
            cv2.line(overlay, (700, 650), (700, 810), (0, 0, 0), 2)
            cv2.rectangle(overlay, (280, 490), (325, 650), (0, 0, 0), -1)
            cv2.rectangle(overlay, (400, 490), (355, 650), (0, 0, 0), -1)
            cv2.rectangle(overlay, (520, 490), (565, 650), (0, 0, 0), -1)
            cv2.rectangle(overlay, (600, 490), (645, 650), (0, 0, 0), -1)
            cv2.rectangle(overlay, (675, 490), (720, 650), (0, 0, 0), -1)
            frame1 = cv2.addWeighted(overlay, 0.4, frame1, 1, 0)
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                    finger1_x = int(handLandmarks.landmark[4].x * 1000)
                    finger1_y = int(handLandmarks.landmark[4].y * 1000)


                    if handLandmarks.landmark[4].y > handLandmarks.landmark[5].y:
                        if 220 < finger1_x < 300 and 650 < finger1_y < 820:
                            gesture_text = 'do'
                            sound_dir = r'.\piano\p_do.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                        elif 280 < finger1_x < 325 and 650 > finger1_y > 490:
                            gesture_text = 'do#'
                            sound_dir = r'.\piano\p_dos.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                        elif 300 < finger1_x < 380 and 650 < finger1_y < 820:
                            gesture_text = 're'
                            sound_dir = r'.\piano\p_re.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                        elif 355 < finger1_x < 400 and 650 > finger1_y > 490:
                            gesture_text = 're#'
                            sound_dir = r'.\piano\p_res.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()

                        elif 380 < finger1_x < 460 and 650 < finger1_y < 820:
                            gesture_text = 'mi'
                            sound_dir = r'.\piano\p_mi.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()

                        elif 460 < finger1_x < 540 and 650 < finger1_y < 820:
                            gesture_text = 'pa'
                            sound_dir = r'.\piano\p_fa.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                        elif 520 < finger1_x < 565 and 650 > finger1_y > 490:
                            gesture_text = 'pa#'
                            sound_dir = r'.\piano\p_fas.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()

                        elif 540 < finger1_x < 620 and 650 < finger1_y < 820:
                            gesture_text = 'sol'
                            sound_dir = r'.\piano\p_sol.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                        elif 600 < finger1_x < 645 and 650 > finger1_y > 490:
                            gesture_text = 'sol#'
                            sound_dir = r'.\piano\p_sols.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()

                        elif 620 < finger1_x < 700 and 650 < finger1_y < 820:
                            gesture_text = 'ra'
                            sound_dir = r'.\piano\p_ra.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                        elif 675 < finger1_x < 720 and 650 > finger1_y > 490:
                            gesture_text = 'ra#'
                            sound_dir = r'.\piano\p_ras.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()

                        elif 700 < finger1_x < 780 and 650 < finger1_y < 820:
                            gesture_text = 'ti'
                            sound_dir = r'.\piano\p_si.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()

        if control == 'd':
            gesture_text = 'Drum'
            cv2.line(overlay, (0, 350), (1000, 350), (0, 0, 0), 2)
            frame1 = cv2.addWeighted(overlay, 0.4, frame1, 1, 0)
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                    finger1_x = int(handLandmarks.landmark[4].x * 1000)
                    finger1_y = int(handLandmarks.landmark[4].y * 1000)

                    if 0 < finger1_x < 400 and 0 < finger1_y < 400:
                        gesture_text = 'leftcymbal'
                        sound_dir = r'.\drum\leftcymbal.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                    elif 450 < finger1_x < 550 and 0 < finger1_y < 400:
                        gesture_text = 'lefttang'
                        sound_dir = r'.\drum\lefttang.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                    elif 600 < finger1_x < 700 and 0 < finger1_y < 400:
                        gesture_text = 'righttang'
                        sound_dir = r'.\drum\righttang.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                    elif 750 < finger1_x < 1500 and 0 < finger1_y < 400:
                        gesture_text = 'rightcybal'
                        sound_dir = r'.\drum\rightcymbal.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                    elif 0 < finger1_x < 350 and 500 < finger1_y < 1000:
                        gesture_text = 'leftdrum'
                        sound_dir = r'.\drum\leftdrum.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                    elif 400 < finger1_x < 650 and 500 < finger1_y < 1000:
                        gesture_text = 'base'
                        sound_dir = r'.\drum\base.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                    elif 700 < finger1_x < 1000 and 500 < finger1_y < 1000:
                        gesture_text = 'rightdrum'
                        sound_dir = r'.\drum\rightdrum.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()

        cv2.putText(frame1, text='name: {}'.format(gesture_text)
                    , org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)


        cv2.imshow("Frame", frame1)
        key = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(1) & 0xFF == ord('p'):
            control = 'p'
            print('p')
        if cv2.waitKey(1) & 0xFF == ord('r'):
            control = 'r'
            print('r')
        if cv2.waitKey(1) & 0xFF == ord('d'):
            control = 'd'
            print('d')
        if key == ord("q"):
            stream.stop_stream()
            stream.close()
            p.terminate()
            break
