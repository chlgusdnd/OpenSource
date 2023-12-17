import pygame
import time

# Pygame 초기화
pygame.init()

# Mixer 초기화
pygame.mixer.init()


# 오디오 파일 딕셔너리
if(play_instrument == 'r'):
    audio_mapping = {
        'c': r'.\recorder\do.wav',
        'd': r'.\recorder\re.wav',
        'e': r'.\recorder\mi.wav',
        'f': r'.\recorder\fa.wav',
        'g': r'.\recorder\sol.wav',
        'a': r'.\recorder\la.wav',
        'b': r'.\recorder\si.wav',
        'h': r'.\recorder\highdo.wav',
    }
elif(play_instrument == 'p'):
    audio_mapping = {
            'c': r'.\piano\p_do.wav',
            'd': r'.\piano\p_re.wav',
            'e': r'.\piano\p_mi.wav',
            'f': r'.\piano\p_fa.wav',
            'g': r'.\piano\p_sol.wav',
            'a': r'.\piano\p_ra.wav',
            'b': r'.\piano\p_si.wav',
            'h': r'.\piano\p_highdo.wav',
        }


play_instrument = input("어떤 악기로 연주해볼까요?(리코더: r, 피아노: p): ")
song_speed = int(input("배속을 입력해주세요: "))
# 파일 경로 설정
file_path = r'.\testplayer.txt'

# 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        for char in line:
            # 각 글자에 대한 오디오 파일 확인 및 재생
            if char in audio_mapping:
                audio_file = audio_mapping[char]
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                time.sleep(1/song_speed)  # 재생 시간을 조절할 수 있습니다.
                pygame.mixer.music.stop()
            else:
                print(f'Undefined character: {char}')

# Pygame 종료
pygame.quit()
