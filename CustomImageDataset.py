import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as tvio
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image according to ImageNET
            transforms.Resize((64, 64))
        ])
        self.classes = self._find_classes(root_dir)
        self.samples = self._make_dataset(root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = self.read_image_as_tensor(img_path)  # Load image as tensor
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, class_name

    def _find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        return classes

    def _make_dataset(self, directory):
        images = []
        for target_class in self.classes:
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, target_class)  
                    images.append(item)
        return images

    def read_image_as_tensor(self, img_path):
        image = tvio.read_image(img_path)
        if image.shape[0] == 4:  # PNG image sometimes has 4 channels.
            image = image[:3, :, :]  # Keep only the first 3 channels
        return image

if __name__ == '__main__':
    image_folder = 'DL Cep Data'

    # Create dataset
    custom_dataset = CustomImageDataset(root_dir=image_folder)
    dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

    # Display sample images with labels
    for images, labels in dataloader:
        # Convert images from tensors to numpy arrays
        images_np = images.permute(0, 2, 3, 1).numpy() 
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(3):
            class_name = labels[i]  # Get the class name directly
            axes[i].imshow(images_np[i])
            axes[i].set_title(f"Label: {class_name}")  # Display the actual class name
            axes[i].axis('off')
        plt.show()
        break
