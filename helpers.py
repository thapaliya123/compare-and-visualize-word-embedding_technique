import os 

def delete_file(file_path):
    # delete existing files
    print('Deleting existing files!!!')
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
            print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")