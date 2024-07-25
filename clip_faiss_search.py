import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image
import os


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def encode_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy()


def create_faiss_index(image_folder):
    image_features = []
    image_paths = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            feature = encode_image(image_path)
            image_features.append(feature)
            image_paths.append(image_path)
    
    image_features = np.concatenate(image_features)
    
    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    
    faiss.write_index(index, "image_index.faiss")
    np.save("image_paths.npy", image_paths)
    
    return index, image_paths


def search_images(query_text, index, image_paths, top_k=2):
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    text_features = text_features.cpu().numpy()
    
    D, I = index.search(text_features, top_k)
    
    return [image_paths[i] for i in I[0]]


if __name__ == "__main__":
    image_folder = "/home/minh/Desktop/docs/pipeline_hcm_ai2/pic"
    
    index, image_paths = create_faiss_index(image_folder)
    
    # Hoặc, nếu đã có index:
    # index = faiss.read_index("image_index.faiss")
    # image_paths = np.load("image_paths.npy").tolist()
    
    query = "a glass"
    results = search_images(query, index, image_paths)
    
    print(f"Top results for '{query}':")
    for path in results:
        print(path)