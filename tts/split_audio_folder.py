import os
import shutil

audio_compressed_dir = 'audio_compressed'
audio_splitted_dir = 'audio_splitted'
max_files_per_folder = 10000

if not os.path.exists(audio_splitted_dir):
    os.makedirs(audio_splitted_dir)

m4a_files = [f for f in os.listdir(audio_compressed_dir) if f.endswith('.m4a')]
total_files = len(m4a_files)
num_folders = (total_files // max_files_per_folder) + (1 if total_files % max_files_per_folder != 0 else 0)

for i in range(num_folders):
    folder_path = os.path.join(audio_splitted_dir, f'folder_{i}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    start_index = i * max_files_per_folder
    end_index = min(start_index + max_files_per_folder, total_files)
    for j in range(start_index, end_index):
        src_file = os.path.join(audio_compressed_dir, m4a_files[j])
        dst_file = os.path.join(folder_path, m4a_files[j])
        shutil.copy(src_file, dst_file)
