from transformers import pipeline

# 사용할 수 있는 TTS 모델 목록
available_models = [
    "facebook/fastspeech2-en-ljspeech",
    "espnet/kan-bayashi_ljspeech_tacotron2",
    "espnet/kan-bayashi_ljspeech_fastspeech2",
    "espnet/kan-bayashi_ljspeech_vits",
]

# 모델을 사용하여 텍스트를 음성으로 변환하는 함수
def text_to_speech_example(model_name):
    tts = pipeline("text-to-speech", model=model_name)
    text = "안녕하세요, 원하는 목소리로 텍스트를 음성으로 변환합니다."
    audio = tts(text)

    with open("output.wav", "wb") as f:
        f.write(audio["audio"])
    print(f"TTS 변환 완료, 모델: {model_name}, output.wav 파일에 저장됨")

# 실행 함수
def main():
    for model_name in available_models:
        try:
            text_to_speech_example(model_name)
            break  # 모델이 성공적으로 로드되면 종료
        except Exception as e:
            print(f"모델 로드 실패: {model_name}, 오류: {e}")

if __name__ == "__main__":
    main()
