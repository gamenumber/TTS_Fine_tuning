import os
from pathlib import Path
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# 모델을 사용하여 텍스트를 음성으로 변환하는 함수
def text_to_speech(text, output_dir):
    # 캐시된 모델 사용
    cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
    
    # SpeechT5 모델과 프로세서 로드
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=cache_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir=cache_dir)
    
    # SpeechT5HifiGan vocoder 로드
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=cache_dir)

    # 텍스트 처리
    inputs = processor(text=text, return_tensors="pt")

    # 스피커 임베딩 생성 (랜덤 사용, 실제로는 특정 화자의 임베딩을 사용할 수 있습니다)
    speaker_embeddings = torch.randn(1, 512)

    # 음성 생성
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # 결과 저장
    output_path = output_dir / "speecht5_tts_with_hifigan_output.wav"
    torchaudio.save(str(output_path), speech.unsqueeze(0), sample_rate=16000)
    print(f"TTS 변환 완료, 저장 위치: {output_path}")

# 실행 함수
def main():
    # 데이터 및 결과 디렉토리 설정
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    result_dir = current_dir / "result"
    result_dir.mkdir(exist_ok=True)
    
    # 텍스트 파일 읽기
    try:
        with open(data_dir / "input_text.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print("입력 텍스트 파일을 찾을 수 없습니다. 기본 텍스트를 사용합니다.")
        text = "annyeonghaseyo gimhyeonbin-ibnida"
    
    try:
        text_to_speech(text, result_dir)
    except Exception as e:
        print(f"TTS 변환 실패, 오류: {e}")

if __name__ == "__main__":
    main()