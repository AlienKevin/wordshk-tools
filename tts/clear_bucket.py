import oss2
from get_bucket import get_bucket
from tqdm import tqdm

bucket = get_bucket()

# List and delete .m4a files
def clear_files():
    # List all files in the bucket
    files_to_delete = [obj.key for obj in oss2.ObjectIterator(bucket)]
    print(f'Total files to delete: {len(files_to_delete)}')
    
    # Delete all files with tqdm
    for file_key in tqdm(files_to_delete):
        result = bucket.delete_object(file_key)
        if result.status != 204:
            print(f'Failed to delete: {file_key}')

# Run the function to delete .m4a files
clear_files()
