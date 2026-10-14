from io import BytesIO
from unittest.mock import patch

from django.contrib.auth.hashers import check_password
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image

from app.models import User


class AccountFlowTests(TestCase):
    def registration_data(self, **overrides):
        data = {
            'username': 'review-user',
            'password': 'long-demo-password',
            'email': 'review@example.com',
            'age': '24',
            'gender': 'Other',
            'phone': '+911234567890',
            'address': 'Test address',
        }
        data.update(overrides)
        return data

    def test_registration_hashes_password_and_login_works(self):
        response = self.client.post('/save/', self.registration_data())
        self.assertRedirects(response, '/login/')
        user = User.objects.get(email='review@example.com')
        self.assertTrue(check_password('long-demo-password', user.password))
        self.assertNotEqual('long-demo-password', user.password)

        response = self.client.post('/loginCheck/', {
            'username': 'review-user', 'password': 'long-demo-password',
        })
        self.assertRedirects(response, '/home/')
        self.assertEqual(self.client.session['useremail'], user.email)

    def test_duplicate_email_and_bad_login_show_errors(self):
        User.objects.create(
            firstname='existing', password='legacy', email='review@example.com',
            age=24, gender='Other', phone='123', address='Address',
        )
        response = self.client.post('/save/', self.registration_data())
        self.assertContains(response, 'already exists')
        response = self.client.post('/save/', self.registration_data(username='existing', email='different@example.com'))
        self.assertContains(response, 'username is already in use', status_code=200)
        response = self.client.post('/loginCheck/', {
            'username': 'missing', 'password': 'wrong',
        })
        self.assertContains(response, 'Invalid username or password')

    def test_legacy_plaintext_password_is_upgraded_after_login(self):
        user = User.objects.create(
            firstname='legacy-user', password='old-demo-password',
            email='legacy@example.com', age=28, gender='Other',
            phone='123', address='Address',
        )
        response = self.client.post('/loginCheck/', {
            'username': 'legacy-user', 'password': 'old-demo-password',
        })
        self.assertRedirects(response, '/home/')
        user.refresh_from_db()
        self.assertTrue(check_password('old-demo-password', user.password))


class ClassifierFlowTests(TestCase):
    def make_image_upload(self):
        image_data = BytesIO()
        Image.new('RGB', (64, 64), (20, 80, 40)).save(image_data, format='PNG')
        return SimpleUploadedFile('satellite.png', image_data.getvalue(), content_type='image/png')

    @override_settings(ALLOWED_HOSTS=['testserver'])
    def test_image_prediction_is_rendered_in_browser(self):
        model = type('Model', (), {'predict': lambda self, data, verbose=0: [[0.9, 0.1, 0, 0, 0, 0, 0]]})()
        with patch('app.views._load_model', return_value=model):
            response = self.client.post('/graph/', {'myfile': self.make_image_upload()})
        self.assertContains(response, 'Landslide')
        self.assertContains(response, '90.0%')
        self.assertContains(response, 'data:image/jpeg;base64,')
        self.assertContains(response, 'Uploaded image used for this prediction')
        self.assertContains(response, 'background-image: linear-gradient')

    def test_missing_and_invalid_images_are_rejected_cleanly(self):
        response = self.client.post('/graph/')
        self.assertContains(response, 'Choose an image to classify')
        response = self.client.post('/graph/', {
            'myfile': SimpleUploadedFile('not-image.txt', b'not an image', content_type='text/plain'),
        })
        self.assertContains(response, 'Upload a valid JPG')

    def test_unsupported_model_output_is_not_misreported(self):
        model = type('Model', (), {'predict': lambda self, data, verbose=0: [[0, 0, 0, 0.9, 0.1, 0, 0]]})()
        with patch('app.views._load_model', return_value=model):
            response = self.client.post('/graph/', {'myfile': self.make_image_upload()})
        self.assertContains(response, 'unsupported class')


class NavigationTests(TestCase):
    def test_public_pages_and_legacy_algorithm_routes_work(self):
        for url in ('/', '/login/', '/reg/', '/home/', '/nvb/'):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)
        for url in ('/pac/', '/svm/', '/dec/', '/randomf/', '/mnb/'):
            with self.subTest(url=url):
                self.assertRedirects(self.client.get(url), '/nvb/')
