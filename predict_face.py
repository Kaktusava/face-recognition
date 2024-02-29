import os
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import onnxruntime as ort

def process_media(model, source, save, save_txt, conf, line_thickness, name, exist_ok, device):
    """
    Process images or a video using the YOLO model.
    """
    # Process each image in the directory
    project = name
    if name[0] == '/':
        project = name
    else:
        name = ''
    results = model(source=source, save=save, save_txt=save_txt, conf=conf, line_width=line_thickness, project=project, name=name, stream=True, exist_ok=exist_ok, half=False, device=device, imgsz=[608, 960])
    for _ in tqdm(results):
        pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process images or video with YOLO model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model file.')
    parser.add_argument('--source', type=str, required=True, help='Directory to search for .jpg files or path to a video file.')
    parser.add_argument('--save', action='store_true', help='Save processed images or video.')
    parser.add_argument('--save_txt', action='store_true', help='Save results to text file.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--line_thickness', type=int, default=1, help='Line thickness for bounding boxes.')
    parser.add_argument('--name', type=str, required=True, help='Output directory for results.')
    parser.add_argument('--exist_ok', action='store_true', help='OK if directory exists.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for inference.')

    args = parser.parse_args()
   
    model = YOLO(args.model_path, task='detect')
    process_media(model, args.source, args.save, args.save_txt, args.conf, args.line_thickness, args.name, args.exist_ok, args.device)