import os
import shutil
from pathlib import Path
import logging
import sys

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ])

def filter_annotated_data(
        source_images_dir: Path,
        source_labels_dir: Path,
        output_filtered_dir: Path,
        source_classes_file: Path
):
    """
    Filters images and labels, keeping only those image-label pairs where a
    corresponding label file exists. Creates a new filtered dataset structure.

    Args:
        source_images_dir (Path): Path to the 'images' folder from Label Studio export.
        source_labels_dir (Path): Path to the 'labels' folder from Label Studio export.
        output_filtered_dir (Path): Directory where the filtered images and labels will be copied.
                                    (e.g., 'filtered_canteen_data/')
        source_classes_file (Path): Path to the 'classes.txt' file from Label Studio export.
                                    This file is copied to the output, but not used for filtering logic.
    """
    if not (source_images_dir.is_dir() and source_labels_dir.is_dir() and source_classes_file.is_file()):
        logging.error(f"Source directories or classes.txt not found. "
                      f"Images: {source_images_dir.exists()}, Labels: {source_labels_dir.exists()}, "
                      f"Classes: {source_classes_file.exists()}")
        sys.exit(1)

    # Create output directories for filtered data
    (output_filtered_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_filtered_dir / "labels").mkdir(parents=True, exist_ok=True)

    logging.info(f"Created filtered output directory at: {output_filtered_dir}")

    # Get all label files
    label_files = [f for f in source_labels_dir.iterdir() if f.is_file() and f.suffix.lower() == '.txt']

    copied_count = 0

    logging.info("Starting filtering process...")
    for label_path in label_files:
        # Extract the original frame name from the label filename
        # Example: '0a5581f7-frame_00100.txt' -> 'frame_00100'
        # We need to find the last occurrence of 'frame_'
        try:
            # Find the part after the last UUID and before .txt
            parts = label_path.stem.split('-')
            if len(parts) > 1:
                image_base_name = '-'.join(parts[1:]) # Rejoin if 'frame_00100' itself had hyphens
            else:
                image_base_name = label_path.stem # If no UUID prefix, assume direct frame name

            # Construct the expected image filename (assuming .jpg)
            image_filename = f"{image_base_name}.jpg"
            source_image_path = source_images_dir / image_filename

            if source_image_path.is_file():
                # Copy image to filtered images directory
                shutil.copy2(source_image_path, output_filtered_dir / "images" / image_filename)

                # Copy label to filtered labels directory
                shutil.copy2(label_path, output_filtered_dir / "labels" / label_path.name)
                copied_count += 1
            else:
                logging.warning(f"Skipping label '{label_path.name}': Corresponding image '{image_filename}' not found.")
        except Exception as e:
            logging.error(f"Error processing label file {label_path.name}: {e}", exc_info=True)

    # Copy classes.txt to the filtered output directory
    shutil.copy2(source_classes_file, output_filtered_dir / source_classes_file.name)

    logging.info(f"Filtering complete. Copied {copied_count} image-label pairs.")
    logging.info(f"Filtered data available at: {output_filtered_dir}")

if __name__ == "__main__":
    # --- Configuration for this filtering script ---

    # Path to the 'images' folder from your Label Studio export.
    # Example: Path("/home/aariyan/Downloads/my_project_export/images/")
    SOURCE_IMAGES_FROM_LS_EXPORT = Path("/media/aariyan/PseudoCode/AI/Labelling/canteen_labels/project-1-at-2025-07-07-12-01-dd1d44d3/images")

    # Path to the 'labels' folder from your Label Studio export.
    # Example: Path("/home/aariyan/Downloads/my_project_export/labels/")
    SOURCE_LABELS_FROM_LS_EXPORT = Path("/media/aariyan/PseudoCode/AI/Labelling/canteen_labels/project-1-at-2025-07-07-12-01-dd1d44d3/labels")

    # Path to the 'classes.txt' file from your Label Studio export.
    # Example: Path("/home/aariyan/Downloads/my_project_export/classes.txt")
    SOURCE_CLASSES_FILE_FROM_LS_EXPORT = Path("/media/aariyan/PseudoCode/AI/Labelling/canteen_labels/project-1-at-2025-07-07-12-01-dd1d44d3/classes.txt")

    # The directory where the filtered images and labels will be saved.
    # This will be an intermediate step before the train/val/test split.
    OUTPUT_FILTERED_DATA_DIR = Path("/media/aariyan/PseudoCode/AI/Labelling/canteen_labels/project-1-at-2025-07-07-12-01-dd1d44d3/filtered_data")

    # --- Run the filtering function ---
    try:
        filter_annotated_data(
            source_images_dir=SOURCE_IMAGES_FROM_LS_EXPORT,
            source_labels_dir=SOURCE_LABELS_FROM_LS_EXPORT,
            output_filtered_dir=OUTPUT_FILTERED_DATA_DIR,
            source_classes_file=SOURCE_CLASSES_FILE_FROM_LS_EXPORT
        )
        logging.info("Filtering script finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during filtering: {e}", exc_info=True)
        sys.exit(1)
