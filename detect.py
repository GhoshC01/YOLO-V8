"""CLI: detect number plates in an image and print the OCR text."""

import argparse
import sys
from pathlib import Path

import cv2

import plate_reader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect number plates and run OCR on an image."
    )
    parser.add_argument(
        "--image",
        default="test_car.jpg",
        help="Path to input image (default: test_car.jpg)",
    )
    parser.add_argument(
        "--model",
        default=plate_reader.DEFAULT_MODEL,
        help="Path to trained YOLO weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=plate_reader.DEFAULT_CONF,
        help=f"Detection confidence threshold (default: {plate_reader.DEFAULT_CONF})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=plate_reader.DEFAULT_IMGSZ,
        help=f"Detection input size; larger finds small plates (default: {plate_reader.DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--output",
        default="output_result.jpg",
        help="Path for annotated output image",
    )
    parser.add_argument(
        "--ocr",
        choices=["paddle", "easy"],
        default="paddle",
        help="OCR engine (default: paddle)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Also report boxes whose text does not look like a plate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = Path(args.image)
    if not data.exists():
        print(f"Error: '{args.image}' ছবিটি খুঁজে পাওয়া যায়নি!")
        sys.exit(1)

    img = plate_reader.decode_image(data.read_bytes())
    if img is None:
        print(f"Error: '{args.image}' পড়া যায়নি — ফরম্যাট সাপোর্টেড নয়।")
        sys.exit(1)

    plates = plate_reader.read_plates(
        img,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        engine=args.ocr,
        include_invalid=args.show_all,
    )

    print(f"OCR engine : {args.ocr}")
    for plate in plates:
        flag = "" if plate["valid_format"] else "  (does not match plate format)"
        print("---------------------------------------")
        print(f"Detected Text: {plate['text']}{flag}")
        print(f"Confidence   : {plate['confidence']:.2f}")
        print("---------------------------------------")

    if not plates:
        print("No license plate detected. Try lowering --conf or raising --imgsz.")

    cv2.imwrite(args.output, plate_reader.annotate(img, plates))
    print(f"Result saved as '{args.output}'")


if __name__ == "__main__":
    main()
