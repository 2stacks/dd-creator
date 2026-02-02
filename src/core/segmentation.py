import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoImageProcessor
try:
    from .birefnet_impl.birefnet import BiRefNet
except ImportError:
    # Fallback or handled if file not present during development
    print("Warning: Local BiRefNet implementation not found.")
    BiRefNet = None

import gc

class BiRefNetSegmenter:
    def __init__(self, model_id="ZhengPeng7/BiRefNet"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is None:
            print(f"Loading BiRefNet model: {self.model_id} on {self.device}...")
            try:
                if BiRefNet is None:
                     raise ImportError("Local BiRefNet module is missing.")
                
                # Use local BiRefNet class
                self.model = BiRefNet.from_pretrained(
                    self.model_id,
                    trust_remote_code=False 
                ).to(self.device)
                
                # Load processor from local directory if config exists, else fallback to Hub
                import os
                local_proc_path = os.path.join(os.path.dirname(__file__), "birefnet_impl")
                if os.path.exists(os.path.join(local_proc_path, "preprocessor_config.json")):
                     self.processor = AutoImageProcessor.from_pretrained(local_proc_path)
                else:
                    self.processor = AutoImageProcessor.from_pretrained(
                        self.model_id, 
                        trust_remote_code=True
                    )
                self.model.eval()
                print("BiRefNet model loaded.")
            except Exception as e:
                print(f"Error loading BiRefNet model {self.model_id}: {e}")
                raise e

    def segment(self, image_path, return_transparent=False):
        self.load_model()
        try:
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            original_size = image.size # (width, height)

            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # BiRefNet expects just the image tensor 'x'
                # Ensure input dtype matches model dtype (e.g. float16 on CUDA)
                input_tensor = inputs["pixel_values"]
                # Get the dtype of the first parameter of the model to match input
                model_dtype = next(self.model.parameters()).dtype
                if input_tensor.dtype != model_dtype:
                    input_tensor = input_tensor.to(model_dtype)
                
                outputs = self.model(input_tensor)
                
                # BiRefNet returns a list of tensors, the last one is the final prediction
                if isinstance(outputs, (list, tuple)):
                    logits = outputs[-1]
                elif hasattr(outputs, 'logits'):
                     logits = outputs.logits
                else:
                    logits = outputs
                
                # Resize to original size
                # Interpolate expects float32 usually for stability, but let's check logits dtype
                if logits.dtype != torch.float32:
                    logits = logits.float()
                    
                prediction = torch.nn.functional.interpolate(
                    logits, 
                    size=original_size[::-1], # (height, width)
                    mode='bilinear', 
                    align_corners=False
                )
                prediction = torch.sigmoid(prediction).cpu().numpy()[0][0]
                
            # Convert to binary mask (white = object, black = background)
            binary_mask = (prediction > 0.5).astype(np.uint8) * 255
            mask_image = Image.fromarray(binary_mask)
            
            if return_transparent:
                # Create RGBA image
                transparent_image = image.convert("RGBA")
                transparent_image.putalpha(mask_image)
                return transparent_image
            
            return mask_image
        except Exception as e:
            print(f"BiRefNet Segmentation error: {e}")
            return None

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("BiRefNet model unloaded.")

# Global instance pattern
_global_segmenter = None

def get_segmenter(model_id="ZhengPeng7/BiRefNet"):
    global _global_segmenter
    if _global_segmenter is None:
        _global_segmenter = BiRefNetSegmenter(model_id)
    return _global_segmenter