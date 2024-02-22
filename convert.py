import cv2
import os
import argparse
from utils.helper import GetLogger, Predictor

logger = GetLogger.logger(__name__)
predictor = Predictor()

def segment_image(input_path, output_dir):
    # Load the image
    image = cv2.imread(input_path)

    if image is None:
        logger.error(f"Could not read the image: {input_path}")
        return

    # Process the image using the Predictor class (assuming it exists)
    out_frame, out_frame_seg = predictor.predict(image)

    # Construct the output file path
    input_filename, input_extension = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(output_dir, f"{input_filename}_segmented{input_extension}")

    # Save the segmented image
    cv2.imwrite(output_path, out_frame_seg)
    print(f"Segmented image saved at {output_path}")


def main(input_dir, output_dir):
    # Convert output directory to absolute path relative to the current working directory
    output_dir = os.path.abspath(output_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            segment_image(input_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment images in a directory')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, help='Directory to save segmented images')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

