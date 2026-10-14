"""Run the bundled model against the labeled sample images."""

import os
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'landslides.settings')

import django

django.setup()

import numpy as np
from PIL import Image

from app.views import _load_model


PROJECT_DIR = Path(__file__).resolve().parent
CLASS_LABELS = {0: 'Landslide', 1: 'No landslide'}


def main():
    model = _load_model()
    correct = 0
    total = 0

    for class_dir in sorted((PROJECT_DIR / 'images').iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        expected = int(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            with Image.open(image_path) as image:
                pixels = np.asarray(
                    image.convert('RGB').resize((50, 50)), dtype=np.float32
                )[None, ...] / 255.0
            probabilities = model.predict(pixels, verbose=0)[0]
            predicted = int(np.argmax(probabilities))
            score = float(probabilities[predicted]) * 100
            expected_label = CLASS_LABELS.get(expected, 'Class {}'.format(expected))
            predicted_label = CLASS_LABELS.get(predicted, 'Unsupported class {}'.format(predicted))
            is_correct = predicted == expected
            correct += int(is_correct)
            total += 1
            print('{}: expected={}, predicted={}, score={:.1f}%, {}'.format(
                image_path.relative_to(PROJECT_DIR), expected_label,
                predicted_label, score, 'OK' if is_correct else 'MISMATCH',
            ))

    if not total:
        raise SystemExit('No labeled sample images found under images/<class>/')
    print('Sample accuracy: {}/{} ({:.1f}%)'.format(correct, total, correct / total * 100))


if __name__ == '__main__':
    main()
