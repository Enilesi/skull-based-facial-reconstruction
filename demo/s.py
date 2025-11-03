import os

folder_path = "/mnt/c/Users/miria/Downloads/male skulls"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
image_files.sort()

for i, filename in enumerate(image_files, start=1):
    ext = os.path.splitext(filename)[1]
    new_name = f"{i}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

print("Redenumirea imaginilor a fost finalizată!")
