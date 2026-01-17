"""
VTO (Virtual Try-On) Service using Custom HuggingFace Space via Gradio Client API
Space: Yuhdeptraico102/VTO
"""
import io
import builtins

import logging
import os
import tempfile
import shutil
import contextlib
from PIL import Image
from typing import Tuple, Optional, Any
from io import BytesIO
import base64
from pathlib import Path

logger = logging.getLogger("vto")

# Global variables for Gradio Client
_client = None
_client_initialized = False

# VTO Configuration - Using custom Space
VTO_SPACE = "Yuhdeptraico102/VTO"
HF_TOKEN = os.environ.get("HF_TOKEN", "Thay bang token HF cua ban o day")

def _get_hf_token() -> str:
    """Get HuggingFace token for Gradio Client."""
    return HF_TOKEN

def _init_gradio_client():
    """Initialize Gradio Client for VTO Space."""
    global _client, _client_initialized

    # Nếu đã khởi tạo rồi thì dùng lại
    if _client_initialized and _client is not None:
        return _client

    try:
        from gradio_client import Client

        logger.info(f"[VTO] Connecting to VTO Space: {VTO_SPACE}...")
        token = _get_hf_token()

        # Bịt print() trong lúc khởi tạo Client để tránh UnicodeEncodeError (dấu ✔)
        orig_print = builtins.print
        buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                # Tắt mọi print tạm thời (kể cả dòng "Loaded as API ... ✔")
                builtins.print = lambda *args, **kwargs: None

                # Thử khởi tạo Client - không truyền hf_token vào constructor
                # gradio_client sẽ tự đọc từ environment variable hoặc không cần token nếu Space là public
                # Lưu lại HF_TOKEN gốc (nếu có), rồi set env cho gradio-client
                original_token = os.environ.get("HF_TOKEN")
                if token:
                    os.environ["HF_TOKEN"] = token
                
                try:
                    # Khởi tạo Client - KHÔNG truyền hf_token vào constructor
                    _client = Client(VTO_SPACE)
                finally:
                    # Khôi phục HF_TOKEN env
                    if original_token is not None:
                        os.environ["HF_TOKEN"] = original_token
                    elif "HF_TOKEN" in os.environ:
                        os.environ.pop("HF_TOKEN", None)

            _client_initialized = True
            logger.info(f"[VTO] Successfully connected to {VTO_SPACE}")
            
            # Log API info for debugging
            try:
                api_info = _client.view_api()
                logger.info(f"[VTO] API Info: {api_info}")
            except Exception as e:
                logger.warning(f"[VTO] Could not fetch API info: {e}")
            
            return _client

        finally:
            # Khôi phục print gốc
            builtins.print = orig_print

    except ImportError:
        error_msg = "[VTO] gradio-client not installed. Please install it: pip install gradio-client"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from None

    except Exception as e:
        error_msg = (
            f"[VTO] Failed to connect to VTO Space: {e}\n"
            "Please check:\n"
            "1. HF_TOKEN is valid\n"
            "2. Space is running\n"
            "3. Internet connection is available"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def _save_image_to_temp(image: Image.Image, prefix: str = "vto") -> str:
    """Save PIL Image to temporary file and return path."""
    temp_dir = tempfile.gettempdir()
    # Use mkstemp to create file and close handle immediately
    fd, temp_path = tempfile.mkstemp(
        dir=temp_dir,
        prefix=f"{prefix}_",
        suffix=".jpg"
    )
    try:
        image.save(temp_path, "JPEG", quality=95)
        return temp_path
    finally:
        # Close file descriptor immediately to avoid file lock issues
        os.close(fd)

def _load_image_from_path(image_path: str) -> Image.Image:
    """Load image from file path."""
    return Image.open(image_path).convert("RGB")

def run_virtual_tryon(
    person_image: Image.Image,
    cloth_image: Image.Image,
    num_inference_steps: int = 30,
    guidance_scale: float = 2.5,
    garment_description: Optional[str] = None
) -> Image.Image:
    """
    Run Virtual Try-On using VTO Space via Gradio Client API.
    
    Args:
        person_image: PIL Image of person/pose
        cloth_image: PIL Image of clothing item
        num_inference_steps: Number of diffusion steps (default: 30)
        guidance_scale: Guidance scale for generation (kept for compatibility)
        garment_description: Optional description of the garment (e.g., "a photo of polo shirt")
    
    Returns:
        PIL Image of result
    """
    try:
        # Initialize Gradio Client
        client = _init_gradio_client()
        from gradio_client import handle_file
        
        # Generate garment description if not provided
        if garment_description is None:
            garment_description = "a photo of clothing item"
        
        logger.info("[VTO] Running virtual try-on via VTO Space...")
        
        # Save images to temporary files
        person_temp = None
        cloth_temp = None
        
        try:
            person_temp = _save_image_to_temp(person_image, "person")
            cloth_temp = _save_image_to_temp(cloth_image, "cloth")
            
            logger.info(f"[VTO] Person image saved to: {person_temp}")
            logger.info(f"[VTO] Cloth image saved to: {cloth_temp}")
            
            # Call VTO Space API
            # API interface: predict(person_img, cloth_img, api_name="/vto_interface")
            logger.info("[VTO] Calling VTO Space API...")
            result = client.predict(
                handle_file(person_temp),
                handle_file(cloth_temp),
                api_name="/vto_interface"
            )
            
            # Result is a string path to the output image
            # Format: path string (e.g., "C:\Users\...\image.webp")
            output_path = result if isinstance(result, str) else (result[0] if isinstance(result, (list, tuple)) else str(result))
            
            logger.info(f"[VTO] VTO Space returned result at: {output_path}")
            
            # Load result image
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output image not found at: {output_path}")
            
            result_image = _load_image_from_path(output_path)
            logger.info("[VTO] Virtual try-on completed successfully")
            
            return result_image
            
        finally:
            # Clean up temporary files
            # Note: Close file handles first to avoid Windows file lock issues
            if person_temp and os.path.exists(person_temp):
                try:
                    # Wait a bit for file handles to be released
                    import time
                    time.sleep(0.1)
                    os.unlink(person_temp)
                except Exception as e:
                    logger.warning(f"[VTO] Failed to delete temp file {person_temp}: {e}")
            
            if cloth_temp and os.path.exists(cloth_temp):
                try:
                    import time
                    time.sleep(0.1)
                    os.unlink(cloth_temp)
                except Exception as e:
                    logger.warning(f"[VTO] Failed to delete temp file {cloth_temp}: {e}")
        
    except Exception as e:
        logger.error(f"[VTO] Error in run_virtual_tryon: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"VTO processing failed: {str(e)}") from e

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))
