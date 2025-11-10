import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import argparse
from datetime import datetime
import os

# ----------------------------
# 1️⃣ 환경 설정
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "xyn-ai/anything-v4.0"  # Anything v4.0 (애니메이션/만화 특화)

print(f"Using device: {device}")

# ----------------------------
# 2️⃣ 인자 파싱
# ----------------------------
parser = argparse.ArgumentParser(description="학습된 LoRA 모델로 이미지 생성")
parser.add_argument(
    "--prompt",
    type=str,
    default="mycharacter, black and white manga style, monochrome illustration",
    help="이미지 생성 프롬프트"
)
parser.add_argument(
    "--negative_prompt",
    type=str,
    default="color, colorful, low quality, blurry",
    help="네거티브 프롬프트"
)
parser.add_argument(
    "--lora_path",
    type=str,
    default="my_lora_model",
    help="LoRA 모델 경로"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs",
    help="생성된 이미지 저장 폴더"
)
parser.add_argument(
    "--num_images",
    type=int,
    default=1,
    help="생성할 이미지 개수"
)
parser.add_argument(
    "--steps",
    type=int,
    default=25,
    help="Inference steps (높을수록 품질 향상, 시간 증가)"
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="CFG scale (높을수록 프롬프트 충실도 증가)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="랜덤 시드 (재현성을 위해 사용)"
)

args = parser.parse_args()

# ----------------------------
# 3️⃣ Stable Diffusion 파이프라인 로드
# ----------------------------
print(f"\nLoading Stable Diffusion model from {model_id}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None  # 안전 검사기 비활성화 (속도 향상)
)

# ----------------------------
# 4️⃣ LoRA 모델 로드
# ----------------------------
print(f"Loading LoRA weights from {args.lora_path}...")
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    args.lora_path,
    torch_dtype=torch.float16
)

pipe.to(device)
pipe.unet.eval()

print("Model loaded successfully!")

# ----------------------------
# 5️⃣ 출력 폴더 생성
# ----------------------------
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------------
# 6️⃣ 이미지 생성
# ----------------------------
print(f"\nGenerating {args.num_images} image(s)...")
print(f"Prompt: {args.prompt}")
print(f"Negative prompt: {args.negative_prompt}")
print(f"Steps: {args.steps}, Guidance scale: {args.guidance_scale}")

# 시드 설정 (재현성)
if args.seed is not None:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(f"Seed: {args.seed}")
else:
    generator = None

for i in range(args.num_images):
    print(f"\nGenerating image {i+1}/{args.num_images}...")

    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]

    # 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{i+1}.png"
    output_path = os.path.join(args.output_dir, filename)

    # 이미지 저장
    image.save(output_path)
    print(f"Saved: {output_path}")

print(f"\n✅ All images generated successfully!")
print(f"📁 Output folder: {args.output_dir}")
