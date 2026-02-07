"""Export dataset to various training tool formats."""

import json
import os
import shutil
from math import ceil


def _collect_image_caption_pairs(output_directory):
    """Walk output dir, skip masks/ and _mask. files, read paired .txt captions.

    Returns list of (image_path, caption_text_or_None, basename).
    """
    if not output_directory or not os.path.isdir(output_directory):
        return []

    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    pairs = []

    for root, dirs, files in os.walk(output_directory):
        if "masks" in dirs:
            dirs.remove("masks")
        if "export" in dirs:
            dirs.remove("export")
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
            if "_mask." in file.lower():
                continue
            img_path = os.path.join(root, file)
            basename = os.path.splitext(file)[0]
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            caption = None
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                except Exception:
                    pass
            pairs.append((img_path, caption, basename))

    pairs.sort(key=lambda x: x[0])
    return pairs


def _validate_export_dir(export_dir):
    """Validate and create export directory. Returns error string or empty string."""
    if not export_dir or not export_dir.strip():
        return "Export directory path is required."
    export_dir = export_dir.strip()
    try:
        os.makedirs(export_dir, exist_ok=True)
        return ""
    except Exception as e:
        return f"Failed to create export directory: {e}"


def _copy_image(src, dest):
    """Copy image file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)


def _write_caption(dest_image_path, caption):
    """Write a .txt caption file alongside the image."""
    txt_path = os.path.splitext(dest_image_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(caption)


def _deduplicate_basename(basename, seen):
    """Return a unique basename, appending _1, _2, etc. if needed."""
    if basename not in seen:
        seen.add(basename)
        return basename
    counter = 1
    while f"{basename}_{counter}" in seen:
        counter += 1
    unique = f"{basename}_{counter}"
    seen.add(unique)
    return unique


def export_kohya(output_directory, export_dir, trigger_word, target_steps=200, progress_cb=None):
    """Export dataset in Kohya_ss format.

    Creates <export_dir>/img/<repeats>_<trigger_word>/ with images and .txt captions.

    Returns (status_message, stats_dict).
    """
    pairs = _collect_image_caption_pairs(output_directory)
    if not pairs:
        return "No images found in output directory.", {}

    if not trigger_word or not trigger_word.strip():
        return "Trigger word is required for Kohya_ss export.", {}

    trigger_word = trigger_word.strip()
    err = _validate_export_dir(export_dir)
    if err:
        return err, {}

    num_images = len(pairs)
    repeats = max(1, ceil(target_steps / num_images))
    folder_name = f"{repeats}_{trigger_word}"
    dest_dir = os.path.join(export_dir, "img", folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    images_exported = 0
    captions_exported = 0
    images_without_captions = 0
    seen_basenames = set()

    for i, (img_path, caption, basename) in enumerate(pairs):
        if progress_cb:
            progress_cb(i, num_images, f"Exporting {basename}...")

        unique_basename = _deduplicate_basename(basename, seen_basenames)
        ext = os.path.splitext(img_path)[1]
        dest_path = os.path.join(dest_dir, unique_basename + ext)
        _copy_image(img_path, dest_path)
        images_exported += 1

        if caption:
            _write_caption(dest_path, caption)
            captions_exported += 1
        else:
            images_without_captions += 1

    stats = {
        "images_exported": images_exported,
        "captions_exported": captions_exported,
        "images_without_captions": images_without_captions,
        "repeats": repeats,
        "folder_path": dest_dir,
    }

    status_lines = [
        f"Kohya_ss export complete: {images_exported} images to {dest_dir}",
        f"Repeats: {repeats} (target {target_steps} steps/epoch, {num_images} images)",
        f"Captions: {captions_exported} exported",
    ]
    if images_without_captions:
        status_lines.append(f"Warning: {images_without_captions} images without captions")

    return "\n".join(status_lines), stats


def export_ai_toolkit(output_directory, export_dir, trigger_word, prepend_if_missing=True, progress_cb=None):
    """Export dataset in AI-Toolkit format.

    Flat folder with images + .txt captions. Replaces trigger_word with [trigger]
    in captions. Optionally prepends [trigger] if not found.

    Returns (status_message, stats_dict).
    """
    pairs = _collect_image_caption_pairs(output_directory)
    if not pairs:
        return "No images found in output directory.", {}

    if not trigger_word or not trigger_word.strip():
        return "Trigger word is required for AI-Toolkit export.", {}

    trigger_word = trigger_word.strip()
    err = _validate_export_dir(export_dir)
    if err:
        return err, {}

    num_images = len(pairs)
    images_exported = 0
    captions_exported = 0
    images_without_captions = 0
    replacements_made = 0
    prepends_added = 0
    seen_basenames = set()

    for i, (img_path, caption, basename) in enumerate(pairs):
        if progress_cb:
            progress_cb(i, num_images, f"Exporting {basename}...")

        unique_basename = _deduplicate_basename(basename, seen_basenames)
        ext = os.path.splitext(img_path)[1]
        dest_path = os.path.join(export_dir, unique_basename + ext)
        _copy_image(img_path, dest_path)
        images_exported += 1

        if caption:
            if trigger_word.lower() in caption.lower():
                # Case-insensitive replacement
                import re
                new_caption = re.sub(re.escape(trigger_word), "[trigger]", caption, flags=re.IGNORECASE)
                replacements_made += 1
            elif prepend_if_missing:
                new_caption = f"[trigger], {caption}"
                prepends_added += 1
            else:
                new_caption = caption
            _write_caption(dest_path, new_caption)
            captions_exported += 1
        else:
            images_without_captions += 1

    stats = {
        "images_exported": images_exported,
        "captions_exported": captions_exported,
        "images_without_captions": images_without_captions,
        "replacements_made": replacements_made,
        "prepends_added": prepends_added,
    }

    status_lines = [
        f"AI-Toolkit export complete: {images_exported} images to {export_dir}",
        f"Captions: {captions_exported} exported ({replacements_made} trigger replacements, {prepends_added} prepends)",
    ]
    if images_without_captions:
        status_lines.append(f"Warning: {images_without_captions} images without captions")

    return "\n".join(status_lines), stats


def export_onetrainer(output_directory, export_dir, format_mode="flat", trigger_word="", target_steps=200, progress_cb=None):
    """Export dataset in OneTrainer format.

    format_mode="kohya" delegates to export_kohya().
    format_mode="flat" copies images + .txt to a flat directory.

    Returns (status_message, stats_dict).
    """
    if format_mode == "kohya":
        return export_kohya(output_directory, export_dir, trigger_word, target_steps, progress_cb)

    pairs = _collect_image_caption_pairs(output_directory)
    if not pairs:
        return "No images found in output directory.", {}

    err = _validate_export_dir(export_dir)
    if err:
        return err, {}

    num_images = len(pairs)
    images_exported = 0
    captions_exported = 0
    images_without_captions = 0
    seen_basenames = set()

    for i, (img_path, caption, basename) in enumerate(pairs):
        if progress_cb:
            progress_cb(i, num_images, f"Exporting {basename}...")

        unique_basename = _deduplicate_basename(basename, seen_basenames)
        ext = os.path.splitext(img_path)[1]
        dest_path = os.path.join(export_dir, unique_basename + ext)
        _copy_image(img_path, dest_path)
        images_exported += 1

        if caption:
            _write_caption(dest_path, caption)
            captions_exported += 1
        else:
            images_without_captions += 1

    stats = {
        "images_exported": images_exported,
        "captions_exported": captions_exported,
        "images_without_captions": images_without_captions,
    }

    status_lines = [
        f"OneTrainer export complete: {images_exported} images to {export_dir}",
        f"Captions: {captions_exported} exported",
    ]
    if images_without_captions:
        status_lines.append(f"Warning: {images_without_captions} images without captions")

    return "\n".join(status_lines), stats


def export_huggingface(output_directory, export_dir, caption_key="text", progress_cb=None):
    """Export dataset in HuggingFace format.

    Flat folder with images + single metadata.jsonl file.
    Each JSONL line: {"file_name": "img.jpg", "<caption_key>": "caption"}

    Returns (status_message, stats_dict).
    """
    pairs = _collect_image_caption_pairs(output_directory)
    if not pairs:
        return "No images found in output directory.", {}

    if not caption_key or not caption_key.strip():
        caption_key = "text"
    caption_key = caption_key.strip()

    err = _validate_export_dir(export_dir)
    if err:
        return err, {}

    num_images = len(pairs)
    images_exported = 0
    captions_exported = 0
    images_without_captions = 0
    metadata_entries = []
    seen_basenames = set()

    for i, (img_path, caption, basename) in enumerate(pairs):
        if progress_cb:
            progress_cb(i, num_images, f"Exporting {basename}...")

        unique_basename = _deduplicate_basename(basename, seen_basenames)
        ext = os.path.splitext(img_path)[1]
        dest_filename = unique_basename + ext
        dest_path = os.path.join(export_dir, dest_filename)
        _copy_image(img_path, dest_path)
        images_exported += 1

        entry = {"file_name": dest_filename}
        if caption:
            entry[caption_key] = caption
            captions_exported += 1
        else:
            entry[caption_key] = ""
            images_without_captions += 1
        metadata_entries.append(entry)

    # Write metadata.jsonl
    jsonl_path = os.path.join(export_dir, "metadata.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats = {
        "images_exported": images_exported,
        "captions_exported": captions_exported,
        "images_without_captions": images_without_captions,
        "metadata_file": jsonl_path,
    }

    status_lines = [
        f"HuggingFace export complete: {images_exported} images to {export_dir}",
        f"Metadata: {jsonl_path} ({len(metadata_entries)} entries)",
        f"Captions: {captions_exported} exported (key: \"{caption_key}\")",
    ]
    if images_without_captions:
        status_lines.append(f"Warning: {images_without_captions} images without captions")

    return "\n".join(status_lines), stats


def push_to_huggingface(export_dir, repo_name, token=None, private=True, progress_cb=None):
    """Push an exported dataset directory to the HuggingFace Hub.

    Returns (status_message, repo_url_or_empty_string).
    """
    from huggingface_hub import HfApi, get_token as hf_get_token

    if not export_dir or not os.path.isdir(export_dir):
        return "Export directory not found. Please export your dataset first.", ""

    # Check directory has files
    has_files = any(
        f for _, _, files in os.walk(export_dir) for f in files
    )
    if not has_files:
        return "Export directory is empty. Please export your dataset first.", ""

    if not repo_name or not repo_name.strip():
        return "Repository name is required.", ""
    repo_name = repo_name.strip()

    # Resolve token
    resolved_token = token.strip() if token and token.strip() else None
    if not resolved_token:
        resolved_token = hf_get_token()
    if not resolved_token:
        return (
            "Not authenticated. Paste a token from https://huggingface.co/settings/tokens, "
            "run 'huggingface-cli login', or set the HF_TOKEN environment variable."
        ), ""

    try:
        if progress_cb:
            progress_cb("Authenticating...")
        api = HfApi(token=resolved_token)
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{repo_name}"

        if progress_cb:
            progress_cb(f"Creating repository {repo_id}...")
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        if progress_cb:
            progress_cb(f"Uploading to {repo_id}...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=export_dir,
            repo_type="dataset",
            commit_message="Upload dataset via Diffusion Dataset Creator",
        )

        visibility = "private" if private else "public"
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        return f"Pushed to {repo_url} ({visibility})", repo_url

    except Exception as e:
        return f"Push failed: {e}", ""
