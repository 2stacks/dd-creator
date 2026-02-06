from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class ProjectState:
    source_directory: str = ""
    output_directory: str = ""
    image_paths: List[str] = field(default_factory=list)
    captions: dict = field(default_factory=dict)  # path -> caption
    masks: dict = field(default_factory=dict)  # path -> mask_path
    upscaled: dict = field(default_factory=dict)  # source_path -> output_path
    transparent: dict = field(default_factory=dict)  # source_path -> transparent_path

    def get_processing_status(self, image_path: str) -> dict:
        """Get processing status for an image."""
        return {
            "has_caption": image_path in self.captions and bool(self.captions[image_path]),
            "has_mask": image_path in self.masks,
            "has_transparent": image_path in self.transparent,
            "has_upscaled": image_path in self.upscaled,
        }
    
    def get_output_path(self, source_path: str, ext: str) -> str:
        """
        Determines the output path for a given source image and extension.
        If output_directory is set, maintains relative structure.
        Otherwise, saves alongside the source file.
        """
        if not self.output_directory:
            return os.path.splitext(source_path)[0] + ext
            
        # Compute relative path from source root
        try:
            rel_path = os.path.relpath(source_path, self.source_directory)
        except ValueError:
            # Fallback if paths are on different drives or inconsistent
            rel_path = os.path.basename(source_path)
            
        base_rel = os.path.splitext(rel_path)[0]
        out_full = os.path.join(self.output_directory, base_rel + ext)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_full), exist_ok=True)
        return out_full

    def scan_directory(self, directory: str, output_directory: str = ""):
        if not os.path.isdir(directory):
            return "Invalid source directory", 0, 0
        
        self.source_directory = directory
        self.output_directory = output_directory
        self.image_paths = []
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    
                    # Try to load existing caption
                    # Check output directory first if set, then source
                    caption_found = False

                    # 1. Check Output Dir
                    if self.output_directory:
                        txt_path = self.get_output_path(img_path, ".txt")
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, "r", encoding="utf-8") as f:
                                    self.captions[img_path] = f.read().strip()
                                caption_found = True
                            except: pass

                    # 2. Check Source Dir (if not found in output)
                    if not caption_found:
                        txt_path_src = os.path.splitext(img_path)[0] + ".txt"
                        if os.path.exists(txt_path_src):
                            try:
                                with open(txt_path_src, "r", encoding="utf-8") as f:
                                    self.captions[img_path] = f.read().strip()
                                caption_found = True
                            except:
                                self.captions[img_path] = ""
                        else:
                            self.captions[img_path] = ""

        self.image_paths.sort()
        output_count = 0
        source_only_count = 0
        for p in self.image_paths:
            if not self.captions.get(p):
                continue
            # Check if output dir has this caption
            if self.output_directory:
                txt_path = self.get_output_path(p, ".txt")
                if os.path.exists(txt_path):
                    output_count += 1
                    continue
            # Must have come from source
            source_only_count += 1
        return f"Found {len(self.image_paths)} images.", output_count, source_only_count

# Global instance
global_state = ProjectState()
