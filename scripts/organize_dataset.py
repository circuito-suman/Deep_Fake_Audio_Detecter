import os
import shutil
import glob
from tqdm import tqdm
import argparse

def organize_dataset(fake_source, real_source, target_dir):
    """
    Moves/Copies audio files from complex source structures into 
    simple data/raw/real and data/raw/fake directories.
    """
    
    # Define targets
    real_target = os.path.join(target_dir, "real")
    fake_target = os.path.join(target_dir, "fake")
    
    os.makedirs(real_target, exist_ok=True)
    os.makedirs(fake_target, exist_ok=True)
    
    # --- Process Real Data ---
    if real_source and os.path.exists(real_source):
        print(f"Scanning Real Data at: {real_source}")
        # LJSpeech structure is usually LJSpeech-1.1/wavs/*.wav
        real_files = []
        for root, dirs, files in os.walk(real_source):
            for file in files:
                if file.endswith(".wav"):
                    real_files.append(os.path.join(root, file))
        
        print(f"Found {len(real_files)} real files. Copying...")
        for src in tqdm(real_files):
            dst = os.path.join(real_target, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    else:
        print("Warning: Real source directory not provided or does not exist.")

    # --- Process Fake Data ---
    if fake_source and os.path.exists(fake_source):
        print(f"Scanning Fake Data at: {fake_source}")
        # WaveFake structure usually has subfolders like 'ljspeech_melgan', etc.
        fake_files = []
        for root, dirs, files in os.walk(fake_source):
            for file in files:
                if file.endswith(".wav"):
                    fake_files.append(os.path.join(root, file))
                    
        print(f"Found {len(fake_files)} fake files. Copying...")
        for src in tqdm(fake_files):
            dst = os.path.join(fake_target, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    else:
        print("Warning: Fake source directory not provided or does not exist.")
        
    print("\nOrganization Complete!")
    print(f"Real Files in {real_target}: {len(os.listdir(real_target))}")
    print(f"Fake Files in {fake_target}: {len(os.listdir(fake_target))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize Audio Datasets")
    parser.add_argument("--fake", help="Path to the extracted WaveFake folder")
    parser.add_argument("--real", help="Path to the extracted LJSpeech (Real) folder")
    parser.add_argument("--target", default="data/raw", help="Target directory (default: data/raw)")
    
    args = parser.parse_args()
    
    organize_dataset(args.fake, args.real, args.target)
