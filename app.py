import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import io
from torchvision import transforms
from torchvision.models import efficientnet_b0
from facenet_pytorch import MTCNN

torch.set_grad_enabled(False)

device = torch.device('cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_efficientnet(checkpoint_path):
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    m.to(device)
    m.eval()
    return m

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Only the face-swap detector loads for this deployment — the GAN-image
# detector is documented separately with its own results in the README/notebook,
# to keep this live demo's memory footprint within free-tier limits.
faceswap_model = load_efficientnet(os.path.join(BASE_DIR, 'models', 'best_model_faceswap.pth'))
FAKE_IDX, REAL_IDX = 0, 1

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


def predict_image(input_image):
    img_np = np.array(input_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped_face = detect_and_crop_face(img_bgr)
    face_for_model = Image.fromarray(cropped_face) if cropped_face is not None else input_image.convert('RGB')
    img_tensor = eval_transform(face_for_model).unsqueeze(0).to(device)

    output = faceswap_model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]
    fake_p, real_p = probs[FAKE_IDX].item(), probs[REAL_IDX].item()
    pred = 'FAKE' if fake_p > real_p else 'REAL'
    conf = max(fake_p, real_p)

    face_preview = np.array(face_for_model)
    label = f'{pred} ({conf*100:.1f}% confidence) | fake={fake_p:.3f}, real={real_p:.3f}'
    return face_preview, label


def predict_video(video_file):
    cap = cv2.VideoCapture(video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / 1))

    timestamps, fake_probs = [], []
    frame_idx = 0
    max_frames = 10

    while cap.isOpened() and len(timestamps) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            face = detect_and_crop_face(frame)
            if face is not None:
                face_tensor = eval_transform(Image.fromarray(face)).unsqueeze(0).to(device)
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
    label = f'{verdict} (avg fake probability: {avg_fake*100:.1f}%, {len(fake_probs)} frames analyzed)'

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timestamps, fake_probs, marker='o', color='crimson')
    ax.axhline(0.5, linestyle='--', color='gray', label='Decision threshold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Fake probability')
    ax.set_title(f'Frame-by-Frame Confidence — Verdict: {verdict}')
    ax.set_ylim(0, 1)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    chart_img = np.array(Image.open(buf).convert('RGB'))
    return chart_img, label


with gr.Blocks(title='Face-Swap DeepFake Detector') as demo:
    gr.Markdown('# Face-Swap DeepFake Detector\nTrained on FaceForensics++ (DeepFakes + FaceSwap methods). *Note: this live demo runs the face-swap model only, due to free-tier memory limits. A second GAN-generated image detector (99% accuracy) is documented separately in the project README/notebook.*')

    with gr.Tab('Image'):
        img_input = gr.Image(type='pil', label='Upload a face image')
        face_preview = gr.Image(type='numpy', label='Detected Face')
        img_text = gr.Textbox(label='Verdict')
        img_button = gr.Button('Analyze Image')
        img_button.click(fn=predict_image, inputs=img_input, outputs=[face_preview, img_text])

    with gr.Tab('Video'):
        vid_input = gr.Video(label='Upload a video')
        vid_viz = gr.Image(type='numpy', label='Confidence Over Time')
        vid_text = gr.Textbox(label='Verdict')
        vid_button = gr.Button('Analyze Video')
        vid_button.click(fn=predict_video, inputs=vid_input, outputs=[vid_viz, vid_text])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    demo.launch(server_name='0.0.0.0', server_port=port)
