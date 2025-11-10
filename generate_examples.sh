#!/bin/bash

# 학습된 LoRA 모델로 이미지 생성 예제 스크립트

echo "=== LoRA 이미지 생성 예제 ==="
echo ""

# 예제 1: 기본 설정으로 1장 생성
echo "1. 기본 설정으로 생성..."
python inference.py

echo ""
echo "---"
echo ""

# 예제 2: 커스텀 프롬프트로 생성
echo "2. 커스텀 프롬프트로 생성..."
python inference.py \
    --prompt "mycharacter, action pose, black and white manga style" \
    --num_images 1

echo ""
echo "---"
echo ""

# 예제 3: 고품질 설정 (steps 증가)
echo "3. 고품질 설정으로 생성..."
python inference.py \
    --prompt "mycharacter, detailed portrait, black and white manga style" \
    --steps 50 \
    --guidance_scale 8.0 \
    --num_images 1

echo ""
echo "---"
echo ""

# 예제 4: 동일한 이미지 재생성 (seed 고정)
echo "4. Seed를 고정하여 재현 가능한 이미지 생성..."
python inference.py \
    --prompt "mycharacter, black and white manga style" \
    --seed 42 \
    --num_images 1

echo ""
echo "✅ 모든 예제 완료!"
echo "📁 생성된 이미지는 outputs/ 폴더에서 확인하세요"
