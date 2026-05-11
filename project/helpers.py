import torch
from PIL import Image
from torchvision import transforms


# NOTE: Do not change the order or content of this list.
class_names = [
    "No Finding",
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


eval_transforms = transforms.Compose([
    # Input (str): image file path
    # Output (torch.Tensor): a preprocessed image tensor to feed into the model
    ##### YOUR CODE START #####
    transforms.Lambda(lambda path: Image.open(path)),



    ##### YOUR CODE END #####
])


def load_model():
    """
    Loads and returns a pretrained PyTorch model.

    Returns:
        model (torch.nn.Module): A PyTorch model
            - forward input: An image tensor preprocessed using `eval_transforms`
            - forward output: A logits tensor of shape `(batch_size, num_classes)`,
                              where the class order corresponds to the labels defined in `class_names`
    """
    ##### YOUR CODE START #####




    ##### YOUR CODE END #####
    return model

