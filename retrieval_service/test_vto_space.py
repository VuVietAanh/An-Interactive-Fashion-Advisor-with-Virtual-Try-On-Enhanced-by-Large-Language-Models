"""
Script để test và inspect API interface của VTO Space
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from gradio_client import Client
from PIL import Image
import tempfile

VTO_SPACE = "Yuhdeptraico102/VTO"
HF_TOKEN = os.environ.get("HF_TOKEN", "Thay bang token HF cua ban o day")

def test_vto_space():
    """Test và inspect VTO Space API interface."""
    print("=" * 60)
    print(f"Testing VTO Space: {VTO_SPACE}")
    print("=" * 60)
    
    # Set HF_TOKEN if available
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
    
    try:
        print("\n[1] Connecting to Space...")
        client = Client(VTO_SPACE)
        print("✓ Connected successfully!")
        
        print("\n[2] Inspecting API interface...")
        try:
            api_info = client.view_api()
            print("\nAPI Information:")
            print("-" * 60)
            print(api_info)
            print("-" * 60)
            
            # Try to parse API info
            if isinstance(api_info, dict):
                print("\nAvailable endpoints:")
                for endpoint, info in api_info.items():
                    print(f"  - {endpoint}: {info}")
        except Exception as e:
            print(f"⚠ Could not fetch API info: {e}")
        
        print("\n[3] Testing with dummy images...")
        # Create dummy test images
        person_img = Image.new("RGB", (512, 768), color="lightblue")
        cloth_img = Image.new("RGB", (512, 512), color="red")
        
        person_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cloth_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        
        person_img.save(person_temp.name, "JPEG")
        cloth_img.save(cloth_temp.name, "JPEG")
        
        print(f"  Person image: {person_temp.name}")
        print(f"  Cloth image: {cloth_temp.name}")
        
        from gradio_client import handle_file
        
        # Try different API call patterns
        print("\n[4] Testing API call patterns...")
        
        # Pattern 1: IDM-VTON style
        print("\n  Pattern 1: IDM-VTON style (with input_dict)")
        try:
            input_dict = {
                "background": handle_file(person_temp.name),
                "layers": [],
                "composite": None
            }
            result = client.predict(
                input_dict,
                garm_img=handle_file(cloth_temp.name),
                garment_des="a photo of shirt",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            print(f"  ✓ Success! Result: {result}")
            print(f"    Result type: {type(result)}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Pattern 2: Simple positional
        print("\n  Pattern 2: Simple positional (person, cloth)")
        try:
            result = client.predict(
                handle_file(person_temp.name),
                handle_file(cloth_temp.name),
                api_name="/predict"
            )
            print(f"  ✓ Success! Result: {result}")
            print(f"    Result type: {type(result)}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Pattern 3: Named parameters
        print("\n  Pattern 3: Named parameters (person_image, cloth_image)")
        try:
            result = client.predict(
                person_image=handle_file(person_temp.name),
                cloth_image=handle_file(cloth_temp.name),
                api_name="/predict"
            )
            print(f"  ✓ Success! Result: {result}")
            print(f"    Result type: {type(result)}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Pattern 4: Try without api_name (use default)
        print("\n  Pattern 4: Without api_name (default endpoint)")
        try:
            result = client.predict(
                handle_file(person_temp.name),
                handle_file(cloth_temp.name)
            )
            print(f"  ✓ Success! Result: {result}")
            print(f"    Result type: {type(result)}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Cleanup - close files first
        person_temp.close()
        cloth_temp.close()
        import time
        time.sleep(0.1)  # Wait for handles to be released
        try:
            os.unlink(person_temp.name)
            os.unlink(cloth_temp.name)
        except Exception as e:
            print(f"  ⚠ Could not delete temp files: {e}")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_vto_space()

