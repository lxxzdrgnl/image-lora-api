import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ 환경 설정
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "xyn-ai/anything-v4.0"  # Anything v4.0 (v4.5도 포함, 애니메이션/만화 특화)

print(f"Using device: {device}")

# ----------------------------
# 2️⃣ Stable Diffusion 컴포넌트 로드
# ----------------------------
print("Loading Stable Diffusion components...")

# VAE, UNet, Text Encoder 개별 로드
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# VAE와 Text Encoder는 freeze (학습하지 않음)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Gradient checkpointing으로 메모리 절약
unet.enable_gradient_checkpointing()

# 모델을 device로 이동
vae.to(device)
text_encoder.to(device)
unet.to(device)

print("Components loaded successfully!")

# ----------------------------
# 3️⃣ LoRA 설정
# ----------------------------
print("Setting up LoRA...")

lora_config = LoraConfig(
    r=8,  # rank 증가로 표현력 향상
    lora_alpha=16,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],  # 더 많은 레이어 학습
    lora_dropout=0.05,
    bias="none",
)

unet = get_peft_model(unet, lora_config)
unet.train()  # 학습 모드

# LoRA 파라미터만 학습
trainable_params = [p for p in unet.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

# ----------------------------
# 4️⃣ 데이터셋 로드 및 전처리
# ----------------------------
dataset_path = "./dataset"
image_files = [
    os.path.join(dataset_path, f)
    for f in os.listdir(dataset_path)
    if f.endswith((".png", ".jpg", ".jpeg", ".webp"))
]

print(f"Found {len(image_files)} images in dataset")

def load_and_preprocess_image(img_path, size=512):
    """이미지를 로드하고 VAE latent로 변환"""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device, dtype=torch.float16)

def encode_prompt(prompt_text):
    """텍스트 프롬프트를 인코딩"""
    text_input = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings

# 프롬프트 생성 (흑백 만화 캐릭터)
def generate_prompt():
    """이미지에 대한 프롬프트 생성 - 더 specific하게"""
    # 빈 프롬프트로 학습 (unconditional training)
    # LoRA가 이미지 자체의 특징을 더 잘 학습하도록
    return ""

def compute_snr(timesteps, noise_scheduler):
    """
    Min-SNR weighting을 위한 SNR 계산
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # SNR 계산
    snr = (alpha / sigma) ** 2
    return snr

# ----------------------------
# 5️⃣ 학습 설정 (대폭 개선)
# ----------------------------
num_epochs = 30  # 더 많은 epoch (작은 데이터셋에 필요)
learning_rate = 1e-5  # Learning rate 대폭 낮춤 (더 안정적)
gradient_accumulation_steps = 2  # 메모리 절약을 위한 gradient accumulation
max_grad_norm = 1.0  # Gradient clipping (gradient 폭발 방지)
noise_offset = 0.1  # 어두운 이미지 학습 개선
snr_gamma = 5.0  # Min-SNR weighting (None이면 비활성화)

optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

# Learning rate scheduler (warmup + cosine decay)
from diffusers.optimization import get_cosine_schedule_with_warmup
total_steps = len(image_files) * num_epochs // gradient_accumulation_steps
warmup_steps = total_steps // 10  # 10% warmup
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\n{'='*60}")
print(f"Starting training for {num_epochs} epochs...")
print(f"{'='*60}")
print(f"Learning rate: {learning_rate} (with cosine schedule)")
print(f"Warmup steps: {warmup_steps}/{total_steps}")
print(f"Image resolution: 512x512")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Max gradient norm: {max_grad_norm}")
print(f"Noise offset: {noise_offset}")
print(f"Min-SNR gamma: {snr_gamma}")
print(f"Training mode: Unconditional (empty prompt)")
print(f"{'='*60}")

# Loss 기록을 위한 리스트
loss_history = []

# ----------------------------
# 6️⃣ 학습 루프
# ----------------------------
global_step = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(image_files, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, img_path in enumerate(progress_bar):
        # 이미지 로드 및 전처리
        pixel_values = load_and_preprocess_image(img_path)

        # VAE로 latent 변환
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # 프롬프트 인코딩
        prompt = generate_prompt()
        encoder_hidden_states = encode_prompt(prompt)

        # Noise 추가 (Diffusion forward process with noise offset)
        noise = torch.randn_like(latents)
        if noise_offset > 0:
            # Noise offset 추가 (어두운/밝은 이미지 학습 개선)
            noise += noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # UNet으로 noise 예측
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Loss 계산 (MSE loss with Min-SNR weighting)
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
        loss = loss.mean([1, 2, 3])  # batch dimension만 남김

        # Min-SNR weighting 적용
        if snr_gamma is not None:
            snr = compute_snr(timesteps, noise_scheduler)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            mse_loss_weights = mse_loss_weights / snr
            loss = loss * mse_loss_weights

        loss = loss.mean()  # 평균 loss
        loss = loss / gradient_accumulation_steps  # gradient accumulation을 위해 나눔

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping (gradient 폭발 방지)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            # Optimizer step
            optimizer.step()
            lr_scheduler.step()  # Learning rate 업데이트
            optimizer.zero_grad()

        # Loss 누적 (실제 loss 값으로)
        actual_loss = loss.item() * gradient_accumulation_steps
        epoch_loss += actual_loss
        loss_history.append(actual_loss)
        global_step += 1

        # Progress bar 업데이트 (loss와 learning rate 표시)
        current_lr = lr_scheduler.get_last_lr()[0]
        progress_bar.set_postfix({"loss": f"{actual_loss:.4f}", "lr": f"{current_lr:.2e}"})

        # 메모리 정리 (메모리 부족 방지)
        if global_step % 10 == 0:
            torch.cuda.empty_cache()

    avg_loss = epoch_loss / len(image_files)
    print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")

# ----------------------------
# 7️⃣ 최종 모델 저장
# ----------------------------
print("\nSaving final model...")
output_dir = "my_lora_model"
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Loss 그래프 저장 (moving average 추가)
print("\nSaving loss graph...")
plt.figure(figsize=(12, 6))

# 원본 loss (반투명)
plt.plot(loss_history, alpha=0.3, label='Raw Loss', color='blue')

# Moving average (window=20)
window_size = 20
if len(loss_history) >= window_size:
    moving_avg = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(loss_history)), moving_avg, label=f'Moving Average (window={window_size})', color='red', linewidth=2)

plt.title('Training Loss Over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=150)
print("Loss graph saved to training_loss.png")

# Loss 통계 출력
print(f"\nTraining Statistics:")
print(f"Total steps: {len(loss_history)}")
print(f"Initial loss: {loss_history[0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Average loss: {np.mean(loss_history):.4f}")
print(f"Min loss: {np.min(loss_history):.4f}")
print(f"Max loss: {np.max(loss_history):.4f}")

# ----------------------------
# 8️⃣ 테스트: 학습된 LoRA로 이미지 생성
# ----------------------------
print("\nGenerating test image...")

# Inference를 위한 파이프라인 재구성
unet.eval()
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    torch_dtype=torch.float16
)
pipe.to(device)

# 이미지 생성 (빠른 테스트를 위해 25 steps 사용)
test_prompt = "mycharacter, black and white manga style, monochrome illustration, high quality"
with torch.no_grad():
    image = pipe(test_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

# 결과 저장
output_path = "output_test.png"
image.save(output_path)
print(f"Test image saved to {output_path}")

print("\nTraining completed successfully!")
