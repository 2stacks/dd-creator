from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class ProjectState:
    source_directory: str = ""
    image_paths: List[str] = field(default_factory=list)
    captions: dict = field(default_factory=dict) # path -> caption
    masks: dict = field(default_factory=dict) # path -> mask_path
    
    def scan_directory(self, directory: str):
        if not os.path.isdir(directory):
            return "Invalid directory"
        
        self.source_directory = directory
        self.image_paths = []
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    
                    # Try to load existing caption
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, "r", encoding="utf-8") as f:
                                self.captions[img_path] = f.read().strip()
                        except:
                            self.captions[img_path] = ""
                    else:
                        self.captions[img_path] = ""
        
        self.image_paths.sort()
        return f"Found {len(self.image_paths)} images."

# Global instance
global_state = ProjectState()
