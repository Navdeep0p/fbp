"""Views for the landslide image classification demo."""

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import json
import logging
from django.conf import settings
from django.contrib.auth.hashers import check_password, identify_hasher, make_password
from django.shortcuts import redirect, render
from django.utils.crypto import constant_time_compare
from PIL import Image, ImageOps, UnidentifiedImageError

from app.models import User

logger = logging.getLogger(__name__)


def home(request):
    return render(request, 'home1.html')


def login(request):
    return render(request, 'loginform.html')


def loginCheck(request):
    if request.method != 'POST':
        return redirect('login')

    username = request.POST.get('username', '').strip()
    password = request.POST.get('password') or request.POST.get('email', '')
    user = User.objects.filter(firstname=username).first()
    authenticated = bool(user and check_password(password, user.password))
    if user and not authenticated:
        try:
            identify_hasher(user.password)
        except ValueError:
            # Upgrade accounts created by the original app, which stored the
            # submitted password directly, as soon as they next sign in.
            if constant_time_compare(password, user.password):
                user.password = make_password(password)
                user.save(update_fields=['password'])
                authenticated = True

    if authenticated:
        request.session['useremail'] = user.email
        return redirect('home')

    return render(request, 'loginform.html', {'error': 'Invalid username or password.'})


def logout(request):
    request.session.pop('useremail', None)
    return redirect('accuracy')


def reg(request):
    return render(request, 'register.html')


def save(request):
    if request.method != 'POST':
        return redirect('reg')

    fields = {
        'firstname': request.POST.get('username', '').strip(),
        'email': request.POST.get('email', '').strip().lower(),
        'age': request.POST.get('age', '').strip(),
        'gender': request.POST.get('gender', '').strip(),
        'phone': request.POST.get('phone', '').strip(),
        'address': request.POST.get('address', '').strip(),
    }
    password = request.POST.get('password', '')
    if not all(fields.values()) or not password:
        return render(request, 'register.html', {
            'error': 'Please complete every field.', 'form_data': request.POST,
        })
    try:
        fields['age'] = int(fields['age'])
        if not 17 <= fields['age'] <= 120:
            raise ValueError
    except ValueError:
        return render(request, 'register.html', {
            'error': 'Enter an age between 17 and 120.', 'form_data': request.POST,
        })
    if User.objects.filter(email=fields['email']).exists():
        return render(request, 'register.html', {
            'error': 'An account with this email already exists.', 'form_data': request.POST,
        })
    if User.objects.filter(firstname=fields['firstname']).exists():
        return render(request, 'register.html', {
            'error': 'That username is already in use.', 'form_data': request.POST,
        })

    User.objects.create(password=make_password(password), **fields)
    return redirect('login')


def nvb(request):
    return render(request, 'pacweb1.html')


def pac(request):
    return redirect('nvb')


def svm(request):
    return redirect('nvb')


def dec(request):
    return redirect('nvb')


def randomf(request):
    return redirect('nvb')


def mnb(request):
    return redirect('nvb')


@lru_cache(maxsize=1)
def _load_model():
    """Load the bundled model once per application process."""
    import tensorflow as tf

    model_path = Path(settings.BASE_DIR)
    with (model_path / 'model.json').open(encoding='utf-8') as model_file:
        architecture = json.load(model_file)
    try:
        model = tf.keras.models.model_from_json(json.dumps(architecture))
    except (TypeError, ValueError):
        # The supplied JSON was exported by Keras 2.2 and uses layer config
        # names removed by Keras 3. Rebuild the same supported layer sequence.
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50, 50, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax'),
        ])
    model.load_weights(str(model_path / 'model.h5'))
    return model


def graph(request):
    if request.method != 'POST':
        return render(request, 'home1.html')

    uploaded = request.FILES.get('myfile')
    if not uploaded:
        return render(request, 'home1.html', {'error': 'Choose an image to classify.'})
    if uploaded.size > 10 * 1024 * 1024:
        return render(request, 'home1.html', {'error': 'Image must be 10 MB or smaller.'})

    try:
        with Image.open(uploaded) as image:
            if image.width * image.height > 50_000_000:
                return render(request, 'home1.html', {
                    'error': 'Image dimensions are too large. Use an image under 50 megapixels.',
                })
            image = ImageOps.exif_transpose(image).convert('RGB')
            preview = image.copy()
            preview.thumbnail((1400, 1000))
            preview_buffer = BytesIO()
            preview.save(preview_buffer, format='JPEG', quality=82, optimize=True)
            image_preview = base64.b64encode(preview_buffer.getvalue()).decode('ascii')
            image = image.resize((50, 50))
            pixels = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        return render(request, 'home1.html', {'error': 'Upload a valid JPG, PNG, or other image file.'})

    try:
        probabilities = _load_model().predict(pixels, verbose=0)[0]
    except Exception:
        logger.exception('Unable to load or run the bundled landslide model')
        return render(request, 'home1.html', {
            'error': 'The prediction model could not be loaded. Check the TensorFlow installation and model files.',
        })

    class_index = int(np.argmax(probabilities))
    # The packaged training folders define class 0 as landslide and class 1
    # as no landslide. Remaining output neurons have no documented labels.
    if class_index >= 2:
        return render(request, 'home1.html', {
            'error': 'This model returned an unsupported class. Please try another image or verify the model training labels.',
        })
    labels = ['Landslide', 'No landslide']
    label = labels[class_index]
    return render(request, 'result.html', {
        'predictedvalue': label,
        'model_score': '{:.1f}%'.format(float(probabilities[class_index]) * 100),
        'image_preview': image_preview,
    })


def accuracy(request):
    return render(request, 'index.html')
