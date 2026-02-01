import torch
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    BlipProcessor, 
    BlipForConditionalGeneration,
    AutoModelForImageClassification,
    AutoImageProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from PIL import Image
import numpy as np
import gc

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
        is_4bit = self.kwargs.get("load_in_4bit", False)
        
        print(f"Loading {self.model_id} (Dtype: {dtype}, 4-bit: {is_4bit})...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        quantization_config = None
        if is_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "vision_model"]
            )
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        self.model.eval()

    def generate_caption(self, image_path, prompt="Write a long descriptive caption for this image in a formal tone.", **kwargs):
        if self.model is None:
            self.load_model()
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error: {e}"

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
        
        # Process inputs
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        
        # Use bfloat16 for pixel values if supported
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
        if "8-bit" in name or "Quantized" in name or "FP8" in name:
            # Use original model with 4-bit NF4 quantization
            return JoyCaptioner("fancyfeast/llama-joycaption-beta-one-hf-llava", load_in_4bit=True)
        return JoyCaptioner("fancyfeast/llama-joycaption-beta-one-hf-llava")
    elif "SmilingWolf" in name or "WD 1.4" in name:
        # Use p1atdev's mirror for WD14 v3 SwinV2 which is compatible with transformers
        return WD14Tagger("p1atdev/wd-swinv2-tagger-v3-hf")
    else:
        # Default
        return Florence2Captioner("microsoft/Florence-2-base")
