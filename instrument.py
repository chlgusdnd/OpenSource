# Import the necessary Packages for this software to run
import time

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
con = 0


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

        ret, frame = cap.read()
        flipped = cv2.flip(frame, flipCode = 3)
        frame1 = cv2.resize(flipped, (1000, 1000))
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        overlay = frame1.copy()
        if control == 'r':
            gesture_text = 'Recoder'
            cv2.line(overlay, (0, 700), (1000, 700), (0, 0, 0), 2)
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                    for point in handsModule.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        # pixelCoordinatesLandmark= drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 1000, 1000)
                        if point == 4:
                            right_x = int(normalizedLandmark.x * 1000)
                            right_y = int(normalizedLandmark.y * 1000)
                            if right_y > 700:
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


                        if True:
                            if finger1_right_y + 175 < finger5_right_y:
                                if finger1_right_x - 30 < finger5_right_x < finger1_right_x + 30:
                                    gesture_text = 'do'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\do.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 225), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 225), (0, 0, 255), 3)

                            elif finger1_right_y + 125 < finger5_right_y:
                                if finger1_right_x - 30 < finger4_right_x < finger1_right_x + 30:
                                    gesture_text = 're'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\re.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x , finger1_right_y + 175), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 175), (0, 0, 255), 3)
                            elif finger1_right_y + 75 < finger5_right_y:
                                if finger1_right_x - 30 < finger3_right_x < finger1_right_x + 30:
                                    gesture_text = 'mi'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\mi.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 125), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 125), (0, 0, 255), 3)
                            elif finger1_right_y + 25 < finger5_right_y:
                                if finger1_right_x - 30 < finger2_right_x < finger1_right_x + 30:
                                    gesture_text = 'pa'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\fa.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 75), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 75), (0, 0, 255), 3)

                            elif finger1_right_y - 25 < finger5_right_y:
                                if finger1_right_x - 30 < finger5_left_x < finger1_right_x + 30:
                                    gesture_text = 'sol'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\sol.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y + 25), (0, 0, 255), 3)
                            elif finger1_right_y - 50 < finger5_right_y:
                                if finger1_right_x - 30 < finger4_left_x < finger1_right_x + 30:
                                    gesture_text = 'la'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\la.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y - 25), (0, 0, 255), 3)
                            elif finger1_right_y - 75 < finger5_right_y:
                                if finger1_right_x - 30 < finger3_left_x < finger1_right_x + 30:
                                    gesture_text = 'ti'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\si.wav'
                                    pygame.mixer.music.load(sound_dir)
                                    pygame.mixer.music.play()
                                    # cv2.rectangle(overlay, (finger1_right_x - 60, finger1_right_y - 175), (finger1_right_x, finger1_right_y + 25), (0,0,255), 3)
                                    cv2.rectangle(frame1, (finger1_right_x - 60, finger1_right_y - 175),
                                                  (finger1_right_x, finger1_right_y - 50), (0, 0, 255), 3)
                            elif finger1_right_y - 125 < finger5_right_y:
                                if finger1_right_x - 30 < finger2_left_x < finger1_right_x + 30:
                                    gesture_text = 'do2'
                                    sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\recorder\highdo.wav'
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
                    finger1_x = int(handLandmarks.landmark[8].x * 1000)
                    finger1_y = int(handLandmarks.landmark[8].y * 1000)
                    if handLandmarks.landmark[7].y < handLandmarks.landmark[8].y:
                        if 200 > finger1_x:
                            con = 1
                            title = input("file name")
                            time.sleep(5)
                            file = open(title, 'w', encoding='utf-8')

                    if handLandmarks.landmark[7].y < handLandmarks.landmark[8].y:
                        if 220 < finger1_x < 300 and 650 < finger1_y < 820:
                            gesture_text = 'do'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_do.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('c')
                        elif 280 < finger1_x < 325 and 650 > finger1_y > 490:
                            gesture_text = 'do#'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_dos.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('c#')
                        elif 300 < finger1_x < 380 and 650 < finger1_y < 820:
                            gesture_text = 're'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_re.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('d')
                        elif 355 < finger1_x < 400 and 650 > finger1_y > 490:
                            gesture_text = 're#'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_res.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('d#')

                        elif 380 < finger1_x < 460 and 650 < finger1_y < 820:
                            gesture_text = 'mi'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_mi.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('e')
                        elif 460 < finger1_x < 540 and 650 < finger1_y < 820:
                            gesture_text = 'pa'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_fa.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('f')
                        elif 520 < finger1_x < 565 and 650 > finger1_y > 490:
                            gesture_text = 'pa#'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_fas.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('f#')
                        elif 540 < finger1_x < 620 and 650 < finger1_y < 820:
                            gesture_text = 'sol'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_sol.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('g')
                        elif 600 < finger1_x < 645 and 650 > finger1_y > 490:
                            gesture_text = 'sol#'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_sols.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('g#')

                        elif 620 < finger1_x < 700 and 650 < finger1_y < 820:
                            gesture_text = 'ra'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_ra.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('a')
                        elif 675 < finger1_x < 720 and 650 > finger1_y > 490:
                            gesture_text = 'ra#'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_ras.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('a#')
                        elif 700 < finger1_x < 780 and 650 < finger1_y < 820:
                            gesture_text = 'ti'
                            sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\piano\p_si.wav'
                            pygame.mixer.music.load(sound_dir)
                            pygame.mixer.music.play()
                            if con == 1:
                                file.write('b')
                        if handLandmarks.landmark[7].y < handLandmarks.landmark[8].y:
                            if 800 < finger1_x:
                                con = 0
                                file.close()


        if control == 'd':
            gesture_text = 'Drum'
            cv2.line(overlay, (0, 350), (1000, 350), (0, 0, 0), 2)
            cv2.line(overlay, (250, 0), (250, 350), (0, 0, 0), 2)
            cv2.line(overlay, (500, 0), (500, 350), (0, 0, 0), 2)
            cv2.line(overlay, (750, 0), (750, 350), (0, 0, 0), 2)
            cv2.line(overlay, (350, 350), (350, 1000), (0, 0, 0), 2)
            cv2.line(overlay, (750, 350), (750, 1000), (0, 0, 0), 2)
            frame1 = cv2.addWeighted(overlay, 0.4, frame1, 1, 0)
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                    finger1_x = int(handLandmarks.landmark[4].x * 1000)
                    finger1_y = int(handLandmarks.landmark[4].y * 1000)

                    if handLandmarks.landmark[7].y < handLandmarks.landmark[8].y:
                        if 200 > finger1_x:
                            con = 1
                            title = input("file name")
                            time.sleep(5)
                            file = open(title, 'w', encoding='utf-8')

                    if 0 < finger1_x < 250 and 0 < finger1_y < 400:
                        gesture_text = 'leftcymbal'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\leftcymbal.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('c')
                    elif 270 < finger1_x < 500 and 0 < finger1_y < 400:
                        gesture_text = 'lefttang'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\lefttang.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('d')
                    elif 520 < finger1_x < 750 and 0 < finger1_y < 400:
                        gesture_text = 'righttang'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\righttang.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('e')
                    elif 770 < finger1_x < 1500 and 0 < finger1_y < 400:
                        gesture_text = 'rightcybal'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\rightcymbal.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('f')
                    elif 0 < finger1_x < 350 and 500 < finger1_y < 1000:
                        gesture_text = 'leftdrum'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\leftdrum.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('g')
                    elif 370 < finger1_x < 650 and 500 < finger1_y < 1000:
                        gesture_text = 'base'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\base.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('a')
                    elif 670 < finger1_x < 1000 and 500 < finger1_y < 1000:
                        gesture_text = 'rightdrum'
                        sound_dir = r'C:\Users\user\PycharmProjects\pythonProject7\OpenSource\drum\rightdrum.wav'
                        pygame.mixer.music.load(sound_dir)
                        pygame.mixer.music.play()
                        file.write('b')
                    if handLandmarks.landmark[7].y < handLandmarks.landmark[8].y:
                        if 800 < finger1_x:
                            con = 0
                            file.close()



        cv2.putText(frame1, text='name, vol, record: {}, {}, {}'.format(gesture_text, vol, con)
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