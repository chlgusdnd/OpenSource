import cv2
import mediapipe as mp

# =========================================

# ===========================================


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 캠 키기
height,width = 576,768
cap = cv2.VideoCapture(0)
ret,image = cap.read()
image = cv2.resize(image,(width,height))

fourcc = cv2.VideoWriter_fourcc('m', 'p','4','v')

# mp_hands의 Hands 정보를 설정하고 읽어들임
with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7,
                    min_tracking_confidence=0.5, max_num_hands=2,) as hands:
    # 캠이 켜져있을때
    while cap.isOpened():
        # 캠 읽기 성공여부 success와 읽은 이미지를 image에 저장
        success, image = cap.read()
        # 캠 읽기 실패시 continue
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        overlay = image.copy()
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.rectangle(overlay, (250, 180), (350, 450), (255, 255, 255), -1)



        # 리코더 그리기----------------------------------------------------------------------------
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


        image = cv2.addWeighted(overlay, 0.4, image, 1, 0)
        gesture_text = 'Cant found hand'

        if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    finger1_x = int(hand_landmarks.landmark[4].x * 100)
                    finger2_x = int(hand_landmarks.landmark[8].x * 100)
                    finger3_x = int(hand_landmarks.landmark[12].x * 100)
                    finger4_x = int(hand_landmarks.landmark[16].x * 100)
                    finger5_x = int(hand_landmarks.landmark[20].x * 100)

                    finger1_y = int(hand_landmarks.landmark[4].y * 100)
                    finger2_y = int(hand_landmarks.landmark[8].y * 100)
                    finger3_y = int(hand_landmarks.landmark[12].y * 100)
                    finger4_y = int(hand_landmarks.landmark[16].y * 100)
                    finger5_y = int(hand_landmarks.landmark[20].y * 100)

                if hand_landmarks.landmark[4].y > hand_landmarks.landmark[5].y:
                    if finger1_x > 16 and finger1_x < 24 and finger1_y > 40:  # 도

                        message = 'C'

                    if finger1_x > 16 and finger1_x < 24 and finger1_y < 40:  # 도#

                        message = 'C#'

                    if finger1_x > 63 and finger1_x < 71 and finger1_y > 40:  # 라

                        message = 'A'

                    if finger1_x > 63 and finger1_x < 71 and finger1_y < 40:  # 라#

                        message = 'A#'

                elif hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y:
                    if finger2_x > 25 and finger2_x < 33 and finger2_y > 40:  # 레

                        message = 'D'

                    if finger2_x > 25 and finger2_x < 33 and finger2_y < 40:  # 레#
                        message = 'D#'
                    if finger2_x > 72 and finger2_x < 81:  # 시

                        message = 'B'

                elif hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y:
                    if finger3_x > 34 and finger3_x < 43:  # 미

                        message = 'E'

                elif hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y:
                    if finger4_x > 44 and finger4_x < 52 and finger4_y > 40:  # 파

                        message = 'F'

                    if finger4_x > 44 and finger4_x < 52 and finger4_y < 40:  # 파#

                        message = 'F#'

                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y:
                    if finger5_x > 53 and finger5_x < 62 and finger5_y > 40:  # 솔

                        message = 'G'

                    if finger5_x > 53 and finger5_x < 62 and finger5_y < 40:  # 솔#

                        message = 'G#'


                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 캠화면에 텍스트를 작성
        cv2.putText(image, text='223428 : {}'.format(gesture_text)
                    , org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)

        # 캠 화면 ( 이미지 )을 화면에 띄움
        cv2.imshow('image', image)

        # q입력시 종료
        if cv2.waitKey(1) == ord('q'):
            break

# 캠 종료
cap.release()
cv2.destroyAllWindows()



