import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

def load_segmentation_model():
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def segment_person(image: Image.Image, model):
    transform = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    segmentation = output.argmax(0).byte().cpu().numpy()
    mask = (segmentation == 15).astype(np.uint8) * 255
    return mask