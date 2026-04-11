import zipfile
from pathlib import Path

def extract_all_zips(source_dir, dest_dir, audio_extensions=(".flac", ".wav", ".mp3", ".ogg")):
    """
    Finds all zip files in a source directory and extracts their audio files 
    into a destination directory.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # 1. Create the destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Check if the source directory actually exists
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: The source directory '{source_path.resolve()}' does not exist.")
        return

    # 3. Grab all .zip files in the source directory
    zip_files = list(source_path.glob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in '{source_path.resolve()}'.")
        return
        
    print(f"Found {len(zip_files)} zip archive(s). Starting extraction...\n")
    
    total_extracted = 0
    
    # 4. Loop through each zip file found
    for zip_file in zip_files:
        print(f"Processing '{zip_file.name}'...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                extracted_from_current = 0
                
                # Check every file inside the current zip
                for file_info in zip_ref.infolist():
                    if file_info.filename.lower().endswith(audio_extensions):
                        # Extract it to the destination
                        zip_ref.extract(file_info, dest_path)
                        extracted_from_current += 1
                        total_extracted += 1
                        
                print(f"  -> Extracted {extracted_from_current} audio files.")
                
        except zipfile.BadZipFile:
            print(f"  -> Error: '{zip_file.name}' is corrupted or not a valid zip file. Skipping.")
            
    print(f"\nAll done! Successfully extracted a total of {total_extracted} audio files to '{dest_path.resolve()}'.")

if __name__ == "__main__":
    # Based on your current location: project/pran_codes
    
    # Go up one level (..), then into Samples
    SOURCE_DIRECTORY = "../Samples" 
    
    # Stay in the current directory (.), and create/use extract_audio
    DESTINATION_DIRECTORY = "./extract_audio" 
    
    extract_all_zips(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, audio_extensions=(".flac", ".wav"))