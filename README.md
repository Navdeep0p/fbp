# Landslide Image Classifier

A Django demo that classifies uploaded satellite images with the bundled CNN model (`model.json` and `model.h5`). Images are resized to 50 × 50 RGB pixels and normalized before prediction.

## Run locally

Use Python 3.9 or newer, activate the project virtual environment, install the pinned dependencies, then run:

```sh
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`, choose **Start classification**, and upload a JPG or PNG image no larger than 10 MB. The bundled sample images are under `images/0` (landslide) and `images/1` (no landslide).

Registration is optional for trying the classifier. New account passwords are stored using Django's password hashing. This project is configured for SQLite locally. To use another database or expose it outside localhost, configure Django settings for that deployment first.

For an HTTPS deployment, set `DJANGO_PRODUCTION=1`, provide a unique random `DJANGO_SECRET_KEY`, set `DJANGO_DEBUG=0`, and set `DJANGO_ALLOWED_HOSTS` to the deployment hostnames. `DJANGO_PRODUCTION=1` enables the secure session and CSRF cookies, SSL redirect, and HSTS settings.

## Model limitation

The model output layer contains seven classes, while the included labeled image folders document only class 0 (landslide) and class 1 (no landslide). The app reports those two classes and displays a message if the model selects an unsupported output. Do not interpret the displayed softmax score as a validated real-world accuracy measure.
