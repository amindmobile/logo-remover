import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from utils.model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)


def remove_logo(pil_image, use_stable_diffusion=False, detection_model_path="models/detection.pth"):
    # Загрузка модели детекции
    detection_model = UNet().to(device)
    detection_model.load_state_dict(torch.load(detection_model_path, map_location=device))
    detection_model.eval()

    # Преобразование изображения
    image_tensor = preprocess(pil_image)

    # Генерация маски
    with torch.no_grad():
        mask = torch.sigmoid(detection_model(image_tensor))
    mask = (mask > 0.5).float().cpu().numpy()[0][0]
    mask = (mask * 255).astype(np.uint8)

    # Конвертация в PIL Image
    original_image = pil_image.resize((256, 256))
    mask_image = Image.fromarray(mask).resize((256, 256))

    # Inpainting
    if use_stable_diffusion and torch.cuda.is_available():
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16
        ).to(device)

        result = pipe(
            prompt="clean background, high quality",
            image=original_image,
            mask_image=mask_image,
        ).images[0]
    else:
        # OpenCV inpainting
        image_np = np.array(original_image)
        result = cv2.inpaint(image_np, mask, 3, cv2.INPAINT_TELEA)
        result = Image.fromarray(result)

    return np.array(result.resize(pil_image.size))