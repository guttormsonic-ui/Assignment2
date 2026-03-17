import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

import torchvision

# --- Oxford-IIIT Pet Dataset ---

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transforms=None):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.image_dir = os.path.join(root, 'images')
        self.annotation_dir = os.path.join(root, 'annotations', 'xmls')

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")

        self.image_files = []
        for f in os.listdir(self.image_dir):
            if f.endswith('.jpg'):
                annotation_file = os.path.splitext(f)[0] + '.xml'
                annotation_path = os.path.join(self.annotation_dir, annotation_file)

                if not os.path.exists(annotation_path):
                    print(f"DEBUG: Skipping image {f}: No corresponding annotation file found at {annotation_path}")
                    continue
                if os.path.getsize(annotation_path) == 0:
                    print(f"DEBUG: Skipping image {f}: Annotation file {annotation_path} is empty")
                    continue

                try:
                    tree = ET.parse(annotation_path)
                    root_xml = tree.getroot()
                    
                    has_valid_objects = False
                    # Check if there are any objects at all
                    objects_in_annotation = root_xml.findall('object')
                    if not objects_in_annotation:
                        print(f"DEBUG: Skipping image {f}: No objects found in annotation {annotation_file}")
                        continue

                    for obj in objects_in_annotation:
                        bndbox = obj.find('bndbox')
                        if bndbox is None:
                            print(f"DEBUG: Object in {annotation_file} has no bndbox. Skipping object.")
                            continue

                        # Basic check for presence of all coordinates
                        coords_str = [bndbox.find(coord) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
                        if not all(c is not None and c.text is not None for c in coords_str):
                            print(f"DEBUG: Object in {annotation_file} has incomplete bndbox coordinates. Skipping object.")
                            continue

                        try:
                            xmin = float(coords_str[0].text)
                            ymin = float(coords_str[1].text)
                            xmax = float(coords_str[2].text)
                            ymax = float(coords_str[3].text)
                        except ValueError:
                            print(f"DEBUG: Invalid numeric coordinate in {annotation_file}. Skipping object.")
                            continue

                        # Ensure xmin < xmax and ymin < ymax for non-degenerate boxes
                        if (xmax - xmin) < 1.0 or (ymax - ymin) < 1.0:
                            print(f"DEBUG: Degenerate bounding box [{xmin},{ymin},{xmax},{ymax}] in {annotation_file}. Skipping object.")
                            continue
                        
                        # Check if the object's class is in the selected classes
                        breed_from_filename = '_'.join(os.path.splitext(f)[0].split('_')[:-1]).title()
                        if breed_from_filename in self.classes:
                            has_valid_objects = True
                        break
                    if has_valid_objects:
                        self.image_files.append(f)
                    else:
                        print(f"DEBUG: Skipping image {f} from {self.image_dir} because no objects with selected classes and valid data were found in {annotation_file}")

                except ET.ParseError as e:
                    print(f"DEBUG: Malformed XML file skipped: {annotation_file} Error: {e}")
                    continue
                except ValueError as e:
                    print(f"DEBUG: Error parsing coordinates in {annotation_file}: {e}. Skipping image {f}.")
                    continue
        
        if not self.image_files:
            print(f"ERROR: OxfordPetDataset initialized but found 0 valid image files.\n  Image directory: {self.image_dir}\n  Annotation directory: {self.annotation_dir}\n  Configured classes: {self.classes}")
        else:
            print(f"DEBUG: OxfordPetDataset initialized with {len(self.image_files)} valid image files.")

    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None: # Skip if no bounding box information
                continue

            xmin_str = bndbox.find('xmin')
            ymin_str = bndbox.find('ymin')
            xmax_str = bndbox.find('xmax')
            ymax_str = bndbox.find('ymax')

            # Ensure all coordinate tags exist and have text
            if not all(c is not None and c.text is not None for c in [xmin_str, ymin_str, xmax_str, ymax_str]):
                # This warning is now more likely caught in __init__ for skipping entire images
                continue

            try:
                xmin = float(xmin_str.text)
                ymin = float(ymin_str.text)
                xmax = float(xmax_str.text)
                ymax = float(ymax_str.text)
            except ValueError:
                # This warning is now more likely caught in __init__
                continue

            # Ensure xmin <= xmax and ymin <= ymax by swapping if necessary
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)

            # Filter out degenerate boxes (width or height <= 0)
            if (xmax - xmin) < 1.0 or (ymax - ymin) < 1.0:
                # This warning is now more likely caught in __init__
                continue

            img_filename = os.path.splitext(os.path.basename(annotation_path))[0]
            breed_from_filename = '_'.join(img_filename.split('_')[:-1]).title()
            try:
                label_idx = self.classes.index(breed_from_filename)
            except ValueError:
                label_idx = 0

            # Only append if label is not background (0) OR if we want to include background objects explicitly.
            # For object detection, we usually only care about the target classes.
            # If label_idx is 0 because the class wasn't in self.classes, we should probably skip this box.
            # If label_idx is 0 because it's genuinely a background class, it depends on model requirements.
            # For now, let's include boxes that correspond to our selected classes.
            if label_idx != 0:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_idx)
            # else:
                # print(f"DEBUG: Skipping object with label {label_idx} (background or unselected class) in {annotation_path}")

        if not boxes:
            # Return dummy data for empty annotations or images with no valid objects
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }

        return {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.xml')

        img = Image.open(img_path).convert("RGB")
        target = self._parse_annotation(annotation_path)

        target["image_id"] = torch.tensor([idx])

        if target['boxes'].numel() > 0:
            areas = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        else:
            areas = torch.zeros((0,), dtype=torch.float32)
        target["area"] = areas

        target["iscrowd"] = torch.zeros((len(target['boxes']),), dtype=torch.int64) # Ensure this matches the number of boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_files)

# --- Penn-Fudan Pedestrian Dataset ---

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs_dir = os.path.join(root, "PNGImages")
        self.masks_dir = os.path.join(root, "PedMasks")

        # Ensure we only process images for which we have corresponding masks
        self.img_mask_pairs = []
        all_imgs = sorted(os.listdir(self.imgs_dir))
        all_masks = sorted(os.listdir(self.masks_dir))

        # Penn-Fudan masks are named like 'FudanPed00001_mask.png' for image 'FudanPed00001.png'
        img_basenames = {os.path.splitext(img_name)[0] for img_name in all_imgs}

        for mask_name in all_masks:
            if mask_name.endswith('_mask.png'):
                base_name = mask_name.replace('_mask.png', '')
                img_name = base_name + '.png'
                if base_name in img_basenames:
                    self.img_mask_pairs.append((img_name, mask_name))

    def __getitem__(self, idx):
        img_name, mask_name = self.img_mask_pairs[idx]
        img_path = os.path.join(self.imgs_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask) # instances are encoded as different colors
        obj_ids = obj_ids[1:] # first id is the background, so remove it

        masks = mask == obj_ids[:, None, None] # split the mask into individual binary masks

        num_objs = len(obj_ids)
        boxes = []
        if num_objs > 0:
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
        else:
            # Handle case where no objects are found in mask
            # This dummy box needs to be valid and not degenerate
            boxes.append([0., 0., 1., 1.]) # Dummy box with positive dimensions
            labels = torch.zeros((1,), dtype=torch.int64) # Background label
            masks = torch.zeros((1, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            num_objs = 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # there is only one class (pedestrian)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_mask_pairs)

# --- Transformations and DataLoader Functions ---

class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            width, height = image.shape[-1], image.shape[-2] # Image is a tensor (C, H, W)

            image = torchvision.transforms.functional.hflip(image)

            if target['boxes'].numel() > 0: # Check if there are any boxes to flip
                bbox = target["boxes"]
                # Flip bounding box coordinates. After this, xmin might be > xmax if original xmax was greater than xmin
                # but the subsequent min/max operation on the _parsed_ coordinates already ensures correct order.
                # The actual flip operation should maintain the ordering internally if coordinates are always xmin,ymin,xmax,ymax.

                # Create new boxes based on the flip transformation
                new_xmin = width - bbox[:, 2]
                new_xmax = width - bbox[:, 0]

                # Ensure the order (xmin, ymin, xmax, ymax) is maintained after flip
                bbox[:, 0] = torch.min(new_xmin, new_xmax)
                bbox[:, 2] = torch.max(new_xmin, new_xmax)

                target["boxes"] = bbox

            if "masks" in target:
                target["masks"] = torchvision.transforms.functional.hflip(target["masks"])

        return image, target

# Custom Compose for Object Detection that handles (image, target) pairs
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_transform(train):
    transforms_list = [ToTensor()]
    if train:
        transforms_list.append(RandomHorizontalFlip(0.5))
    return Compose(transforms_list) # Use our custom Compose


def collate_fn(batch):
    return tuple(zip(*batch))

# Custom letterbox function for YOLOv5 input preprocessing
def letterbox(img_tensor, target, new_shape=(640, 640), fill_value=0.0):
    # img_tensor is C, H, W float tensor [0, 1]
    shape = img_tensor.shape[1:]  # current shape [H, W]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape) # (H_target, W_target)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute new unpadded shape
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # (W_new_unpad, H_new_unpad)
    
    # Calculate padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Resize image
    if shape[::-1] != new_unpad:  # if new_unpad is different from original (W,H)
        img_tensor = torchvision.transforms.Resize(new_unpad[::-1])(img_tensor) # torchvision resize expects (H, W)

    # Pad image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Pad in (left, right, top, bottom) order for H and W dimensions
    # Current img_tensor shape is (C, H_resized, W_resized)
    img_tensor = torch.nn.functional.pad(img_tensor, (left, right, top, bottom), "constant", fill_value)

    # Adjust bounding boxes
    boxes = target['boxes'] # (N, 4) tensor (xmin, ymin, xmax, ymax)
    if boxes.numel() > 0:
        # Scale boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left  # x coordinates
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top   # y coordinates
        
        # Clamp coordinates to be within the new image boundaries (0 to new_shape[1] for x, 0 to new_shape[0] for y)
        boxes[:, 0] = boxes[:, 0].clamp(0, new_shape[1])
        boxes[:, 1] = boxes[:, 1].clamp(0, new_shape[0])
        boxes[:, 2] = boxes[:, 2].clamp(0, new_shape[1])
        boxes[:, 3] = boxes[:, 3].clamp(0, new_shape[0])
        
        target['boxes'] = boxes
    
    return img_tensor, target

# Define a specific collate_fn for YOLOv5
def yolov5_collate_fn(batch, img_size=640):
    images, targets = zip(*batch)
    
    letterboxed_images = []
    letterboxed_targets = []
    
    for img, tgt in zip(images, targets):
        processed_img, processed_tgt = letterbox(img, tgt, new_shape=(img_size, img_size))
        letterboxed_images.append(processed_img)
        letterboxed_targets.append(processed_tgt)
        
    images_batch = torch.stack(letterboxed_images, 0)
    
    return images_batch, letterboxed_targets

def create_dataloader(dataset, batch_size, shuffle, num_workers, collate_fn):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )