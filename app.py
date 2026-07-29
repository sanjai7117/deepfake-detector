import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_b0
from facenet_pytorch import MTCNN

device = torch.device('cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def unnormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return np.clip(img.permute(1, 2, 0).numpy(), 0, 1)

def load_efficientnet(checkpoint_path):
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.to(device)
    m.eval()
    return m

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
gan_model = load_efficientnet(os.path.join(BASE_DIR, 'models', 'best_model.pth'))
faceswap_model = load_efficientnet(os.path.join(BASE_DIR, 'models', 'best_model_faceswap.pth'))
FAKE_IDX, REAL_IDX = 0, 1

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

gan_gradcam = GradCAM(gan_model, gan_model.features[-1])
faceswap_gradcam = GradCAM(faceswap_model, faceswap_model.features[-1])

mtcnn = MTCNN(keep_all=False, device=device)

def detect_and_crop_face(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is None:
        return None
    x1, y1, x2, y2 = [int(b) for b in boxes[0]]
    x1, y1 = max(0, x1), max(0, y1)
    face_crop = frame_rgb[y1:y2, x1:x2]
    return face_crop if face_crop.size > 0 else None


def predict_image_both(input_image):
    img_np = np.array(input_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped_face = detect_and_crop_face(img_bgr)
    face_for_model = Image.fromarray(cropped_face) if cropped_face is not None else input_image.convert('RGB')
    img_tensor = eval_transform(face_for_model).unsqueeze(0).to(device)

    def run_one(model, gradcam):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
        fake_p, real_p = probs[FAKE_IDX].item(), probs[REAL_IDX].item()
        pred = 'FAKE' if fake_p > real_p else 'REAL'
        conf = max(fake_p, real_p)
        pred_idx = FAKE_IDX if pred == 'FAKE' else REAL_IDX

        cam = gradcam.generate(img_tensor, pred_idx)
        original = unnormalize(img_tensor.squeeze(0))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * original + 0.5 * heatmap
        return overlay, f'{pred} ({conf*100:.1f}% confidence) | fake={fake_p:.3f}, real={real_p:.3f}'

    gan_overlay, gan_text = run_one(gan_model, gan_gradcam)
    fs_overlay, fs_text = run_one(faceswap_model, faceswap_gradcam)
    return gan_overlay, gan_text, fs_overlay, fs_text


def predict_video(video_file):
    cap = cv2.VideoCapture(video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / 1))

    timestamps, fake_probs = [], []
    frame_idx = 0
    max_frames = 15

    while cap.isOpened() and len(timestamps) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            face = detect_and_crop_face(frame)
            if face is not None:
                face_tensor = eval_transform(Image.fromarray(face)).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = faceswap_model(face_tensor)
                    p = torch.softmax(out, dim=1)[0][FAKE_IDX].item()
                timestamps.append(frame_idx / video_fps)
                fake_probs.append(p)
        frame_idx += 1
    cap.release()

    if not fake_probs:
        return None, 'No face detected in video.'

    avg_fake = np.mean(fake_probs)
    verdict = 'FAKE' if avg_fake > 0.5 else 'REAL'
    label = f'{verdict} (avg fake probability: {avg_fake*100:.1f}%, {len(fake_probs)} frames analyzed) — Face-Swap Detector'

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timestamps, fake_probs, marker='o', color='crimson')
    ax.axhline(0.5, linestyle='--', color='gray', label='Decision threshold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Fake probability')
    ax.set_title(f'Frame-by-Frame Confidence — Verdict: {verdict}')
    ax.set_ylim(0, 1)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    chart_img = np.array(Image.open(buf).convert('RGB'))
    return chart_img, label


with gr.Blocks(title='DeepFake Detector') as demo:
    gr.Markdown('# DeepFake Detector\nTwo specialized models: one for AI-generated (GAN) portraits, one for real video face-swap forgery.')

    with gr.Tab('Image — Compare Both Models'):
        img_input = gr.Image(type='pil', label='Upload a face image')
        with gr.Row():
            with gr.Column():
                gr.Markdown('### GAN-Image Detector')
                gan_viz = gr.Image(type='numpy', label='Grad-CAM')
                gan_text = gr.Textbox(label='Verdict')
            with gr.Column():
                gr.Markdown('### Face-Swap Detector')
                fs_viz = gr.Image(type='numpy', label='Grad-CAM')
                fs_text = gr.Textbox(label='Verdict')
        img_button = gr.Button('Analyze Image')
        img_button.click(fn=predict_image_both, inputs=img_input, outputs=[gan_viz, gan_text, fs_viz, fs_text])

    with gr.Tab('Video — Face-Swap Analysis'):
        vid_input = gr.Video(label='Upload a video')
        vid_viz = gr.Image(type='numpy', label='Confidence Over Time')
        vid_text = gr.Textbox(label='Verdict')
        vid_button = gr.Button('Analyze Video')
        vid_button.click(fn=predict_video, inputs=vid_input, outputs=[vid_viz, vid_text])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    demo.launch(server_name='0.0.0.0', server_port=port)
