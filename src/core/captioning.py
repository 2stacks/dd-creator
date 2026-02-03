import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModelForImageClassification,
    AutoImageProcessor,
    LlavaForConditionalGeneration,
)
from PIL import Image
import numpy as np
import gc
import os
import csv
try:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
except ImportError:
    ort = None
    hf_hub_download = None

class BaseCaptioner:
    def __init__(self, model_id, device=None, **kwargs):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.kwargs = kwargs

    def load_model(self):
        raise NotImplementedError

    def generate_caption(self, image_path, **kwargs):
        raise NotImplementedError

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            if hasattr(self, 'tags'):
                del self.tags
            
            # clear kwargs/cache if any
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model = None
            self.processor = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Unloaded {self.model_id}")

class ONNXCaptioner(BaseCaptioner):
    def load_model(self):
        if self.model is not None:
            return
        
        if ort is None or hf_hub_download is None:
            raise ImportError("onnxruntime-gpu (or onnxruntime) and huggingface-hub are required for ONNX models.")

        print(f"Loading ONNX model {self.model_id}...")
        
        # Download model and tags
        try:
            model_path = hf_hub_download(repo_id=self.model_id, filename="model.onnx")
            tags_path = hf_hub_download(repo_id=self.model_id, filename="selected_tags.csv")
        except Exception as e:
            raise RuntimeError(f"Failed to download model or tags: {e}")

        # Load Tags
        self.tags = []
        with open(tags_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                self.tags.append(row[1]) # tag name is usually index 1 (tag_id, name, type, count)

        # Load Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.device == "cpu":
            providers = ['CPUExecutionProvider']
            
        self.model = ort.InferenceSession(model_path, providers=providers)
        
        # Get input name and shape
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape[1:3] # (Batch, H, W, 3) -> (H, W) usually

    def preprocess(self, image: Image.Image, height: int, width: int):
        # 1. Resize with padding (keeping aspect ratio) or simple resize?
        # WD14 usually expects simple resize or smart resize. 
        # For simplicity and standard compliance with most taggers: BICUBIC resize to target.
        
        # Ensure RGBA -> RGB -> BGR
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize((width, height), Image.BICUBIC)
        
        # Array conversion
        img_array = np.array(image, dtype=np.float32)
        
        # RGB to BGR
        img_array = img_array[:, :, ::-1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0) 
        
        return img_array

    def generate_caption(self, image_path, threshold=0.35, **kwargs):
        if self.model is None:
            self.load_model()
            
        try:
            image = Image.open(image_path)
        except Exception as e:
            return f"Error: {e}"

        # Target size from model input or default 448
        h, w = 448, 448
        if hasattr(self, 'input_shape') and self.input_shape:
             h, w = self.input_shape
             
        input_tensor = self.preprocess(image, h, w)
        
        # Run inference
        outputs = self.model.run(None, {self.input_name: input_tensor})
        probs = outputs[0][0] # (1, num_tags) -> (num_tags,)
        
        # Post-process tags
        results = []
        # General tags usually start after rating tags (first 4 usually: general, sensitive, questionable, explicit)
        # But for WD14 v3, the CSV has 'category' column. 0=general, 9=rating, 4=character
        # Simple thresholding is usually sufficient.
        
        for i, prob in enumerate(probs):
            if prob > threshold:
                if i < len(self.tags):
                    tag = self.tags[i]
                    # Format
                    tag = tag.replace("_", " ")
                    results.append(tag)
                    
        return ", ".join(results)

class Florence2Captioner(BaseCaptioner):
    def load_model(self):
        if self.model is not None:
            return
        print(f"Loading {self.model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def generate_caption(self, image_path, task_prompt="<MORE_DETAILED_CAPTION>"):
        if self.model is None:
            self.load_model()
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error: {e}"

        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if inputs["pixel_values"].dtype in [torch.float16, torch.float32]:
             inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed.get(task_prompt, generated_text)

class BlipCaptioner(BaseCaptioner):
    def load_model(self):
        if self.model is not None:
            return
        print(f"Loading {self.model_id}...")
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_caption(self, image_path, **kwargs):
        if self.model is None:
            self.load_model()
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error: {e}"

        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out[0], skip_special_tokens=True)

class JoyCaptioner(BaseCaptioner):
    def load_model(self):
        if self.model is not None:
            return

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        print(f"Loading {self.model_id} (dtype: {dtype})...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    def generate_caption(self, image_path, prompt="Write a long descriptive caption for this image in a formal tone.", **kwargs):
        if self.model is None:
            self.load_model()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error: {e}"

        # Get target size from processor and pre-resize to avoid lanczos interpolation error
        # The processor uses lanczos which PyTorch F.interpolate doesn't support
        target_size = getattr(self.processor.image_processor, 'size', {})
        if isinstance(target_size, dict):
            h = target_size.get('height', 384)
            w = target_size.get('width', 384)
        else:
            h = w = target_size if isinstance(target_size, int) else 384

        # Resize to exact target size using PIL (supports all interpolation modes)
        image = image.resize((w, h), Image.BICUBIC)

        # Build the conversation exactly as provided
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Format the conversation
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        
        # Process inputs (do_resize=False since we pre-resized to avoid lanczos error)
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt", do_resize=False).to(self.device)

        # Cast pixel_values to match vision tower dtype
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = self.processor.tokenizer.decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return caption.strip()

class WD14Tagger(BaseCaptioner):
    def load_model(self):
        if self.model is not None:
            return
        print(f"Loading {self.model_id}...")
        # Start with a known compatible model or fallback
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load WD1.4 model {self.model_id}: {e}")
            raise e

    def generate_caption(self, image_path, threshold=0.35, **kwargs):
        if self.model is None:
            self.load_model()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error: {e}"

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Cast if needed
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if inputs.get("pixel_values") is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits[0])
        
        # Get tags
        results = []
        for i, prob in enumerate(probs):
            if prob > threshold:
                tag = self.model.config.id2label[i]
                # Filter out system tags like "rating:safe", "general", etc if desired
                # Usually we want general tags.
                # Common replacements to match danbooru format:
                tag = tag.replace("_", " ")
                results.append(tag)
        
        return ", ".join(results)

def get_captioner(name: str):
    if "Florence" in name:
        mid = "microsoft/Florence-2-large" if "Large" in name else "microsoft/Florence-2-base"
        return Florence2Captioner(mid)
    elif "BLIP-Large" in name:
        return BlipCaptioner("Salesforce/blip-image-captioning-large")
    elif "BLIP-Base" in name:
        return BlipCaptioner("Salesforce/blip-image-captioning-base")
    elif "JoyCaption" in name:
        # BF16 only - quantization via bitsandbytes doesn't work with this model
        return JoyCaptioner("fancyfeast/llama-joycaption-beta-one-hf-llava")
    elif "WD ViT" in name:
        return ONNXCaptioner("SmilingWolf/wd-vit-tagger-v3")
    elif "WD ConvNext" in name:
        return ONNXCaptioner("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    elif "SmilingWolf" in name or "WD 1.4" in name:
        # Use p1atdev's mirror for WD14 v3 SwinV2 which is compatible with transformers
        return WD14Tagger("p1atdev/wd-swinv2-tagger-v3-hf")
    else:
        # Default
        return Florence2Captioner("microsoft/Florence-2-base")
