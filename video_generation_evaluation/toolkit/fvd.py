import torch
import torchvision
from torchvision import transforms
import av
import numpy as np
from tqdm import tqdm
from numpy import cov
from numpy import mean


class I3DFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(I3DFeatureExtractor, self).__init__()
        self.model = torchvision.models.video.r3d_18(pretrained=True)
        self.model.fc = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)

def extract_features(video_path, model, device, transform):
    try:
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()
            img = transform(img)
            frames.append(img)
            if len(frames) == 16:
                break
        if len(frames) < 16:
            while len(frames) < 16:
                frames.append(frames[-1])
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(video_tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def get_dataset_features(video_dir, model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989]),
    ])
    features = []
    for video_file in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_file)
        feature = extract_features(video_path, model, model.device, transform)
        if feature is not None:
            features.append(feature)
    return np.array(features)

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = I3DFeatureExtractor().to(device)
model.eval()

real_video_dir = './FVD/real_videos/architecture'
real_features = get_dataset_features(real_video_dir, model, device)

generated_video_dir = './sampled_videos/cogvideox-5b/architecture'
generated_features = get_dataset_features(generated_video_dir, model, device)

mu_real = mean(real_features, axis=0)
mu_generated = mean(generated_features, axis=0)

sigma_real = cov(real_features, rowvar=False)
sigma_generated = cov(generated_features, rowvar=False)

from scipy.linalg import sqrtm

def calculate_fvd(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fvd

fvd_value = calculate_fvd(mu_real, sigma_real, mu_generated, sigma_generated)
print(f"FVD: {fvd_value}")