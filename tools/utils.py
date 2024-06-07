import zipfile
import os
import dropbox
from tqdm import tqdm
from config import DATA

#DROPBOX_ACCESS_TOKEN = 'sl.B2Qa40k0s6Six4YFi-W8rMbTNsTt9EtEOmMKRjX5HUTMrSiwj4tKKkvz_jYo5MNglLfbMcgu-5qLNZ9T7rh2ksFchn-C0l7Dn3GJEkoEQiiiKJPDH-arRT1HOy37WT7hMBP1fI6zSI-N9WqdL21WomA'
DROPBOX_ACCESS_TOKEN = 'sl.B2uCPC7_hkhLpbuI7mFHrAccqZgQZYoW4jZ50UAI9wXBLXnnr1bsM3I4Vvc1wslsscE9DKq7uyXgsZrfgVu2uKKVNHstKm2HUaDZcH-8UrK6uAeQW6jevf-vNvCqbNc9eRLPOm8IKFgtKY-cqVwIBNU'
if not os.path.exists(DATA):
    os.mkdir(DATA)
        
def unzip_file(zip_path, extract_to):
    """Unzip a file to a specified directory.
    
    Args:
    zip_path (str): The path to the zip file.
    extract_to (str): The directory to extract the files to.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Files extracted to {extract_to}")
    except zipfile.BadZipFile:
        print("Error: The file is not a zip file or it is corrupted.")
    except FileNotFoundError:
        print("Error: The zip file does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_data_from_dropbox(dataset):
    """Download a zipped file from Dropbox to local storage with a progress bar.
    
    Args:
    dataset (str): Name of the dataset to be downloaded and unzipped.
    """
    file_path = f'/{dataset}.zip'
    local_path = os.path.join(DATA, f'{dataset}.zip')
    
    if not os.path.exists(local_path):  # Check if zip file already exists locally
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        try:
            metadata, res = dbx.files_download(path=file_path)
            file_size = metadata.size

            with open(local_path, "wb") as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in res.iter_content(chunk_size=4096):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            print("Download successful! File downloaded to:", local_path)
        except Exception as e:
            print("Error downloading file:", e)
    else:
        print("File already exists:", local_path)
    
    # Unzip after downloading
    unzip_file(local_path, DATA)

    return


