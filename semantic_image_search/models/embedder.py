import torch
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipModel


class EmbeddingExtractor:
    def __init__(self, model_name: str, model_type: str, use_gpu: bool = True):
        """
        Initialize the EmbeddingExtractor with the specified model.

        Parameters:
        model_name (str): The name of the model to load (e.g., "openai/clip-vit-base-patch32", "Salesforce/blip-itm-base-coco").
        model_type (str): The type of the model ("clip" or "blip").
        use_gpu (bool): Whether to use GPU for computation. Defaults to True.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        if model_type == "clip":
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        elif model_type == "blip":
            self.model = BlipModel.from_pretrained(model_name)
            self.processor = BlipProcessor.from_pretrained(model_name)
        else:
            raise ValueError("model_type must be either 'clip' or 'blip'")

        self.model.to(self.device)

    @torch.no_grad()
    def extract_text_embeddings(self, texts, normalize: bool = True):
        """
        Extract text embeddings using the specified model.

        Parameters:
        texts (str or list of str): The input text(s) to embed.
        normalize (bool): Whether to normalize the embeddings. Defaults to True.

        Returns:
        torch.Tensor: The text embeddings.
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        embeddings = self.model.get_text_features(**inputs)
        if normalize:
            embeddings = F.normalize(embeddings)
        return embeddings.tolist()

    @torch.no_grad()
    def extract_image_embeddings(self, images, normalize: bool = True):
        """
        Extract image embeddings using the specified model.

        Parameters:
        images (PIL.Image.Image or list of PIL.Image.Image): The input image(s) to embed.
        normalize (bool): Whether to normalize the embeddings. Defaults to True.

        Returns:
        torch.Tensor: The image embeddings.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        embeddings = self.model.get_image_features(**inputs)
        if normalize:
            embeddings = F.normalize(embeddings)
        return embeddings.tolist()


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    # Example usage for CLIP
    clip_extractor = EmbeddingExtractor("openai/clip-vit-base-patch32", "clip")
    text_embeddings = clip_extractor.extract_text_embeddings(["A photo of a cat", "A photo of a dog"])
    image = Image.open("../../inputs/cat_1.jpeg")
    image_embeddings = clip_extractor.extract_image_embeddings([image])

    print(f"Image and text similarities for Clip: {np.array(image_embeddings)@np.array(text_embeddings).T}")

    # Example usage for BLIP
    blip_extractor = EmbeddingExtractor("Salesforce/blip-itm-base-coco", "blip")
    text_embeddings = blip_extractor.extract_text_embeddings(["A photo of a cat", "A photo of a dog"])
    image_embeddings = blip_extractor.extract_image_embeddings([image])
    print(f"Image and text similarities for Blip: {np.array(image_embeddings)@np.array(text_embeddings).T}")
