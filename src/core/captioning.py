import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModelForImageClassification,
    AutoImageProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from PIL import Image
import numpy as np
import gc
import csv
try:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
except ImportError:
    ort = None
    hf_hub_download = None
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB, compositing RGBA onto white background."""
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')

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
        
        # Ensure RGBA -> RGB (composite onto white) -> BGR
        image = _ensure_rgb(image)
        
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

class Qwen25VLCaptioner(BaseCaptioner):
    def __init__(self, model_id, bits=8, **kwargs):
        super().__init__(model_id, **kwargs)
        self.bits = bits

    def load_model(self):
        if self.model is not None:
            return

        if process_vision_info is None:
            raise ImportError("qwen-vl-utils is required for Qwen2.5-VL. Install with: pip install qwen-vl-utils")

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = None

        if self.bits == 4:
            print(f"Loading {self.model_id} (4-bit quantized)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual"],
            )
        elif self.bits == 8:
            print(f"Loading {self.model_id} (8-bit quantized)...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual"],
            )
        else:
            print(f"Loading {self.model_id} ({dtype})...")

        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)
        load_kwargs = dict(device_map="cuda:0" if torch.cuda.is_available() else "cpu")
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = dtype
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        self.model.eval()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Model loaded. GPU memory allocated: {allocated:.2f} GB")

    def generate_caption(self, image_path, prompt="Describe this image in detail.", **kwargs):
        if self.model is None:
            self.load_model()

        try:
            image = _ensure_rgb(Image.open(image_path))
        except Exception as e:
            return f"Error: {e}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

        # Trim prompt tokens from output
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()


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
            image = _ensure_rgb(Image.open(image_path))
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
            image = _ensure_rgb(Image.open(image_path))
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


class JoyCaptionerQuantized(BaseCaptioner):
    """Quantized JoyCaption using 8-bit bitsandbytes quantization.

    Uses LLM.int8() which has mature support for llm_int8_skip_modules.
    Keeps vision_tower and multi_modal_projector in full precision.

    Requires ~10-12GB VRAM instead of ~17GB for the full BF16 model.
    """

    def load_model(self):
        if self.model is not None:
            return

        print(f"Loading {self.model_id} (8-bit quantized)...")

        # Use 8-bit quantization (LLM.int8()) which has better support
        # for llm_int8_skip_modules than 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=[
                "vision_tower",
                "multi_modal_projector",
                "lm_head",
                # Also try with model. prefix patterns
                "model.vision_tower",
                "model.multi_modal_projector",
            ],
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load with 8-bit quantization
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            # Don't specify torch_dtype - let bitsandbytes handle it
        )
        self.model.eval()

        # Report memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Model loaded. GPU memory allocated: {allocated:.2f} GB")

    def generate_caption(self, image_path, prompt="Write a long descriptive caption for this image in a formal tone.", **kwargs):
        if self.model is None:
            self.load_model()

        try:
            image = _ensure_rgb(Image.open(image_path))
        except Exception as e:
            return f"Error: {e}"

        # Get target size from processor and pre-resize to avoid lanczos interpolation error
        target_size = getattr(self.processor.image_processor, 'size', {})
        if isinstance(target_size, dict):
            h = target_size.get('height', 384)
            w = target_size.get('width', 384)
        else:
            h = w = target_size if isinstance(target_size, int) else 384

        # Resize to exact target size using PIL (supports all interpolation modes)
        image = image.resize((w, h), Image.BICUBIC)

        # Build the conversation
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

        # Process inputs (do_resize=False since we pre-resized)
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt", do_resize=False)

        # Move to device - handle potential dtype issues
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Cast pixel_values to float16 for 8-bit inference
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

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
            image = _ensure_rgb(Image.open(image_path))
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
    if "Qwen" in name and "3B" in name:
        return Qwen25VLCaptioner("Qwen/Qwen2.5-VL-3B-Instruct", bits=16)
    elif "Qwen" in name and "4-bit" in name:
        return Qwen25VLCaptioner("Qwen/Qwen2.5-VL-7B-Instruct", bits=4)
    elif "Qwen" in name:
        return Qwen25VLCaptioner("Qwen/Qwen2.5-VL-7B-Instruct", bits=8)
    elif "BLIP-Large" in name:
        return BlipCaptioner("Salesforce/blip-image-captioning-large")
    elif "BLIP-Base" in name:
        return BlipCaptioner("Salesforce/blip-image-captioning-base")
    elif "JoyCaption" in name:
        model_id = "fancyfeast/llama-joycaption-beta-one-hf-llava"
        if "Quantized" in name or "8-bit" in name:
            return JoyCaptionerQuantized(model_id)
        return JoyCaptioner(model_id)
    elif "WD ViT" in name:
        return ONNXCaptioner("SmilingWolf/wd-vit-tagger-v3")
    elif "WD ConvNext" in name:
        return ONNXCaptioner("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    elif "SmilingWolf" in name or "WD 1.4" in name:
        return WD14Tagger("p1atdev/wd-swinv2-tagger-v3-hf")
    else:
        return BlipCaptioner("Salesforce/blip-image-captioning-base")
