from pathlib import Path
import torch
import soundfile as sf
from TTS.api import TTS

def create_example_voice_file(data_dir):
    sample_rate = 22050
    duration = 1
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t)
    voice_path = data_dir / "voice.wav"
    sf.write(str(voice_path), waveform.numpy(), sample_rate)
    print(f"음성 파일 저장 완료: {voice_path}")

def text_to_speech(text, output_path):
    # 한국어 모델 직접 지정 (adjusted format)
    model_name = "tts_models/ko/univnet/kss"
    
    print(f"선택된 모델: {model_name}")
    
    # TTS 인스턴스 생성 및 모델 로드
    try:
        tts = TTS(model_name, progress_bar=False)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 음성 생성
    try:
        tts.tts_to_file(text=text, file_path=str(output_path))
        print(f"TTS 변환 완료, 저장 위치: {output_path}")
    except Exception as e:
        print(f"TTS 변환 실패: {e}")
        
def main():
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    result_dir = current_dir / "result"
    data_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    create_example_voice_file(data_dir)
    
    try:
        with open(data_dir / "input_text.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print("입력 텍스트 파일을 찾을 수 없습니다. 기본 텍스트를 사용합니다.")
        text = "안녕하세요? 제 이름은 각청입니다."
    
    output_path = result_dir / "tts_output.wav"
    text_to_speech(text, output_path)

if __name__ == "__main__":
    main()