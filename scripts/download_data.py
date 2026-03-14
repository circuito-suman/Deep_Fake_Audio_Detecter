import os
import requests
import zipfile
import tarfile
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASETS = {
    "asvspoof2019_la": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
        "desc": "ASVspoof 2019 Logical Access (TTS & VC)",
        "manual_download": "https://datashare.ed.ac.uk/handle/10283/3336"
    },
    "wavefake": {
        "url": "https://zenodo.org/record/5642694/files/WaveFake.zip?download=1", # Often requires manual download due to traffic limits
        "desc": "WaveFake: A collection of Deepfake Audio (TTS)",
        "manual_download": "https://zenodo.org/record/5642694"
    },
    "in_the_wild": {
        "url": "https://deepfake-demo.aisec.fraunhofer.de/in_the_wild.zip", # Example link, check actual site
        "desc": "In-the-Wild Audio Deepfake Dataset", 
        "manual_download": "https://deepfake-demo.aisec.fraunhofer.de/in_the_wild"
    }
}

def extract_data(file_path, target_dir):
    """Extracts a zip or tar file to the target directory."""
    logger.info(f"Extracting {file_path} to {target_dir}...")
    try:
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
            with tarfile.open(file_path, "r:gz") as tar_ref:
                tar_ref.extractall(target_dir)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return False
        logger.info("Extraction complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to extract: {e}")
        return False

def download_file(url, target_path):
    """Downloads a file from a URL to a target path."""
    logger.info(f"Downloading from {url} to {target_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(50 * downloaded / total_size)
                        print(f"\r[{'=' * percent}{' ' * (50 - percent)}] {downloaded}/{total_size} bytes", end='')
        print() # Newline
        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download or Setup Deepfake Audio Datasets")
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), help='Name of the dataset to download')
    parser.add_argument('--file', type=str, help='Path to a manually downloaded zip file to extract')
    parser.add_argument('--target_dir', type=str, default='data/raw', help='Directory to extract data to')
    
    args = parser.parse_args()
    
    os.makedirs(args.target_dir, exist_ok=True)

    if args.file:
        if os.path.exists(args.file):
            extract_data(args.file, args.target_dir)
        else:
            logger.error(f"File {args.file} not found.")
            
    elif args.dataset:
        info = DATASETS[args.dataset]
        logger.info(f"Selected Dataset: {info['desc']}")
        logger.info(f"URL: {info['url']}")
        logger.warning(f"Note: If automatic download fails, please download manually from: {info['manual_download']}")
        
        filename = os.path.join(args.target_dir, f"{args.dataset}.zip")
        if download_file(info['url'], filename):
            extract_data(filename, args.target_dir)
            # Cleanup zip file (optional)
            # os.remove(filename)
    else:
        print("Please specify a dataset to download or a file to extract.")
        print("Available datasets:")
        for key, val in DATASETS.items():
            print(f"  - {key}: {val['desc']}")
            print(f"    Link: {val['manual_download']}")

if __name__ == "__main__":
    main()
