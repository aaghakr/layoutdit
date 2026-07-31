import gradio as gr
import torch
from PIL import Image, ImageDraw
import numpy as np
from model import design_intent_detector
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import cv2
import matplotlib.pyplot as plt
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def getRegions(design_intent_map, min_area=1000, threshold_ratio=1.0, kernel_n=37, draw=False, draw_img=None):
    """Get design intent regions with area filtering"""
    if draw:
        assert draw_img, "draw_img is required to draw."

    # Threshold using mean value
    threshold_value = design_intent_map.mean() * threshold_ratio
    _, binary = cv2.threshold(design_intent_map, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = (kernel_n, kernel_n)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones(kernel, np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(kernel, np.uint8))
    edges = cv2.Canny(binary, 50, 150)

    # Find contours and filter by area
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if area >= min_area:  # Only keep boxes larger than min_area
            bbox.append((x, y, x+w, y+h))

    if draw:
        draw = ImageDraw.Draw(draw_img)
        for r in bbox:
            draw.rectangle(r, outline="blue", width=5)
        return Image.fromarray(binary), draw_img, bbox
    else:
        return None, None, bbox

def load_model(ckpt_path, device='cuda'):
    """Load the model in the same way as the test script."""
    model = design_intent_detector(act='none', action='forward')
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image_pil(img, target_size=(224, 224)):
    from torchvision import transforms
    pp = get_preprocessing_fn('mit_b1', pretrained='imagenet')
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img)
    transform = transforms.Compose([
        lambda x: cv2.resize(x, (224, 224)),
        pp,
        transforms.ToTensor()
    ])
    image_tensor = transform(arr).float()
    return image_tensor.unsqueeze(0)

def visualize_predictions(original_img, pred_map, boxes):
    """Visualize predictions in the same style as the test pipeline."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax1.imshow(original_img)
    ax1.axis('off')
    ax1.set_title('Input Image')

    # Prediction map
    ax2.imshow(pred_map, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Design Intent Map')

    # Map with bounding boxes
    ax3.imshow(pred_map, cmap='gray')
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=False, edgecolor='blue', linewidth=2)
        ax3.add_patch(rect)
    ax3.axis('off')
    ax3.set_title('Detected Regions')

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def predict(img, ckpt_path, min_area=1000, threshold_ratio=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(ckpt_path, device)
    x = preprocess_image_pil(img).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = pred.squeeze().cpu().numpy()

        # Process prediction map
        pred_img = (pred * 255).astype(np.uint8)
        pred_img_resized = cv2.resize(pred_img, (513, 750))

        # Extract bounding boxes with area filtering
        _, _, boxes = getRegions(pred_img_resized,
                               min_area=min_area,
                               threshold_ratio=threshold_ratio,
                               kernel_n=37,  # Same as test.sh
                               draw=False)

        # Create visualization
        img_resized = img.resize((513, 750))
        viz_img = visualize_predictions(img_resized, pred_img_resized, boxes)

        return viz_img, boxes

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Layout Image"),
        gr.Textbox(label="Model Checkpoint Path",
                  value=str(PROJECT_ROOT / "data/model_weights/intent_map/design_intent_pku_epoch100.pth")),
        gr.Slider(minimum=100, maximum=10000, value=1000, step=100,
                 label="Minimum Box Area (pixels²)"),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                 label="Threshold Ratio (relative to mean)")
    ],
    outputs=[
        gr.Image(type="pil", label="Visualization"),
        gr.JSON(label="Detected Regions (x1, y1, x2, y2)")
    ],
    title="Design Intent Detection Demo",
    description="Upload a layout image and adjust parameters to filter design intent regions. Increase minimum area to remove small detections."
)

def main():
    demo.launch()

if __name__ == "__main__":
    main()
