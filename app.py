import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_image(input_path, output_folder, target_size=(224, 224)):
    try:
        with Image.open(input_path) as img:
            # Convert to RGB mode if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Calculate aspect ratio
            aspect = img.width / img.height
            
            # Calculate new dimensions while maintaining aspect ratio
            if aspect > 1:
                new_width = target_size[0]
                new_height = int(target_size[0] / aspect)
            else:
                new_width = int(target_size[1] * aspect)
                new_height = target_size[1]
            
            # Resize image using high-quality downsampling
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new image with white background
            new_img = Image.new("RGB", target_size, (255, 255, 255))
            
            # Paste resized image onto center of new image
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # Save as PNG
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            new_img.save(output_path, 'PNG')
            
            print(f"Processed: {os.path.basename(input_path)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")

def convert_and_resize_parallel(input_folder, output_folder, target_size=(224, 224)):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for filename in image_files:
            input_path = os.path.join(input_folder, filename)
            futures.append(executor.submit(process_image, input_path, output_folder, target_size))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    img_dirs = [
        r"C:\Users\sneah\Python projects\Oranges\dataset\Object detection train diesease",
    ]

    for i in img_dirs:
        base_path = i.split("\\")
        output_path = r"C:\Users\sneah\Python projects\Oranges\dataset\Object detection train diesease\train" #os.path.join(base_path[0], base_path[1], f"{base_path[2]}_transform")
        convert_and_resize_parallel(i, output_path)
