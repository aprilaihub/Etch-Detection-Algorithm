import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_etchings(image_path: Path, brightness_percentile: float, circularity_threshold: float):
    img, gray = load_image(image_path)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cdf = np.cumsum(hist) / np.sum(hist)
    threshold_idx = np.searchsorted(cdf, brightness_percentile)
    brightness_threshold = int(threshold_idx)

    binary_brightness = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
    contours_bright, _ = cv2.findContours(
        binary_brightness, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    circular_etchings = []
    circularity_values = []
    for cnt in contours_bright:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularity_values.append(circularity)
        if circularity > circularity_threshold:
            circular_etchings.append({
                "contour": cnt,
                "circularity": circularity,
                "area": area,
            })

    return {
        "image_path": image_path,
        "img": img,
        "gray": gray,
        "hist": hist,
        "cdf": cdf,
        "brightness_threshold": brightness_threshold,
        "binary_brightness": binary_brightness,
        "contours_bright": contours_bright,
        "circular_etchings": circular_etchings,
        "circularity_values": circularity_values,
        "circularity_threshold": circularity_threshold,
        "brightness_percentile": brightness_percentile,
    }


def create_test_results_image(image_path: Path, brightness_percentile: float, circularity_threshold: float):
    data = detect_etchings(image_path, brightness_percentile, circularity_threshold)
    img_results = data["img"].copy()

    for etch in data["circular_etchings"]:
        M = cv2.moments(etch["contour"])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img_results, (cx, cy), 32, (0, 255, 0), 4)

    return img_results, data


def create_analysis_report_image(data):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    axes[0, 0].imshow(cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(data["gray"], cmap="gray")
    axes[0, 1].set_title("Grayscale Image", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    img_brightness_filter = cv2.cvtColor(data["binary_brightness"], cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_brightness_filter, data["contours_bright"], -1, (0, 255, 0), 2)
    axes[0, 2].imshow(cv2.cvtColor(img_brightness_filter, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Brightness Filter", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    img_roundness_filter = cv2.cvtColor(data["binary_brightness"], cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_roundness_filter, data["contours_bright"], -1, (255, 0, 0), 2)
    for etch in data["circular_etchings"]:
        cv2.drawContours(img_roundness_filter, [etch["contour"]], -1, (0, 255, 0), 2)
    axes[0, 3].imshow(cv2.cvtColor(img_roundness_filter, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("Roundness Filter", fontsize=12, fontweight="bold")
    axes[0, 3].axis("off")

    img_results = data["img"].copy()
    for etch in data["circular_etchings"]:
        M = cv2.moments(etch["contour"])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img_results, (cx, cy), 32, (0, 255, 0), 4)
    axes[0, 4].imshow(cv2.cvtColor(img_results, cv2.COLOR_BGR2RGB))
    axes[0, 4].set_title("Test Results", fontsize=12, fontweight="bold")
    axes[0, 4].axis("off")

    axes[1, 0].axis("off")
    axes[1, 1].axis("off")

    axes[1, 2].hist(data["gray"].flatten(), bins=256, alpha=0.8, color="skyblue")
    axes[1, 2].axvline(
        x=data["brightness_threshold"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {data['brightness_threshold']}",
    )
    axes[1, 2].set_title("Brightness percentile threshold", fontsize=10, fontweight="bold")
    axes[1, 2].set_xlabel("Pixel Brightness Value")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].legend()

    axes[1, 3].hist(data["circularity_values"], bins=30, alpha=0.8, color="orange")
    axes[1, 3].axvline(
        x=data["circularity_threshold"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {data['circularity_threshold']}",
    )
    axes[1, 3].set_title("Roundness threshold", fontsize=10, fontweight="bold")
    axes[1, 3].set_xlabel("Circularity (0=line, 1=circle)")
    axes[1, 3].set_ylabel("Frequency")
    axes[1, 3].legend()

    total_contours = len(data["contours_bright"])
    total_circular = len(data["circular_etchings"])
    pass_rate = (total_circular / total_contours * 100) if total_contours else 0.0

    summary_text = (
        "DETECTION SUMMARY\n\n"
        f"Total Detections: {total_circular}\n\n"
        "Brightness Filter:\n"
        f"  Threshold: {data['brightness_threshold']}\n"
        f"  Regions found: {total_contours}\n\n"
        "Roundness Filter:\n"
        f"  Threshold: {data['circularity_threshold']}\n"
        f"  Passed: {total_circular}\n"
        f"  Rejected: {total_contours - total_circular}\n\n"
        "Success Rate:\n"
        f"  {pass_rate:.1f}% pass filters\n"
    )

    axes[1, 4].axis("off")
    axes[1, 4].text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def process_image(image_path: Path, output_dir: Path, brightness_percentile: float, circularity_threshold: float):
    output_dir.mkdir(parents=True, exist_ok=True)

    img_results, data = create_test_results_image(
        image_path, brightness_percentile, circularity_threshold
    )

    original_out = output_dir / "original.png"
    results_out = output_dir / "test_results.png"
    analysis_out = output_dir / "analysis.png"

    cv2.imwrite(str(original_out), data["img"])
    cv2.imwrite(str(results_out), img_results)

    fig = create_analysis_report_image(data)
    fig.savefig(str(analysis_out), dpi=200)
    plt.close(fig)

    return {
        "original": original_out,
        "test_results": results_out,
        "analysis": analysis_out,
    }


def collect_images(input_path: Path):
    if input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        return [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]
    return [input_path]


def main():
    parser = argparse.ArgumentParser(description="Etch detection batch runner")
    parser.add_argument("--input", required=True, help="Image path or folder")
    parser.add_argument("--output-dir", default="results", help="Results root folder")
    parser.add_argument("--brightness", type=float, default=0.999, help="Brightness percentile")
    parser.add_argument("--roundness", type=float, default=0.65, help="Circularity threshold")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    images = collect_images(input_path)

    if not images:
        raise ValueError(f"No images found at: {input_path}")

    for image_path in images:
        image_folder = output_root / image_path.stem
        process_image(image_path, image_folder, args.brightness, args.roundness)
        print(f"Processed: {image_path} -> {image_folder}")


if __name__ == "__main__":
    main()
