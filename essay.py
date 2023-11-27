import torch
from models import YOLOModel  # adjust this import to your actual model class and location

def load_model(model_path):
    model = YOLOModel()  # initialize your PyTorch model class
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # set to evaluation mode
    return model

# Path to your model
vehicle_model_path = "/home/amine/Desktop/justep/drive-download-20230912T162408Z-001/Model/best_vehicle_detection.pt"

# Load the vehicle model
model_vehicle = load_model(vehicle_model_path)
