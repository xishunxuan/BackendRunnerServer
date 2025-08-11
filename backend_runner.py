import argparse
import os
import time
from datetime import datetime
# Pillow is used to generate test images
from PIL import Image, ImageDraw, ImageFont

def create_mock_image(output_path: str, text: str) -> None:
    """Generate a simple PNG image with the provided *text* centered.

    This variant is more robust:
    1.  Always falls back to the default bitmap font (eliminates the
        dependency on system fonts such as Arial).
    2.  Correctly handles multiline strings (with ``\n``).
    3.  Catches *all* exceptions and logs a helpful message so the main
        loop can continue even if something goes wrong.
    """

    try:
        img_size = (400, 200)
        bg_color = (255, 255, 255)  # White background
        text_color = (0, 0, 0)      # Black text

        img = Image.new("RGB", img_size, color=bg_color)
        draw = ImageDraw.Draw(img)

        # Always use the built-in default font (platform independent)
        font = ImageFont.load_default()

        # Calculate text bounding box – Pillow changed APIs over time, so we
        # try the newer *multiline_textbbox* first and fall back if needed.
        try:
            bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Older Pillow – use deprecated helpers
            text_width, text_height = draw.multiline_textsize(text, font=font)

        # Center the text
        x = (img_size[0] - text_width) / 2
        y = (img_size[1] - text_height) / 2

        draw.multiline_text((x, y), text, fill=text_color, font=font, align="center")

        # Save PNG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(output_path, f"image_{timestamp}.png")
        img.save(image_path, format="PNG")
        print(f"[backend_runner] Created image: {image_path}")

    except Exception as exc:
        # Catch-all so one failed iteration does not crash the whole runner.
        print(f"[backend_runner] Failed to create image: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Mock backend runner for testing.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--conf_path", type=str, required=True, help="Path to the configuration file.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    # Create / open the log file immediately so monitoring systems can pick
    # it up even before the loop finishes.
    log_path = os.path.join(args.output_path, "final_status.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=============================================\n")
        f.write("Mock Backend Runner Started\n")
        f.write(f"Output Path: {args.output_path}\n")
        f.write(f"Config Path: {args.conf_path}\n")
        f.write("=============================================\n")

        # Simulate a long-running process
        for i in range(3):
            print(f"Running iteration {i + 1}/3 …")
            time.sleep(5)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            create_mock_image(
                args.output_path,
                text=f"Generated at: {now}\nIteration: {i + 1}",
            )

        f.write("Process finished generating images.\n")
        f.write(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print(f"Created final log file: {log_path}")
        print("Mock Backend Runner Finished.")


if __name__ == "__main__":
    main()
