'''
Take picture
============

.. author:: Mathieu Virbel <mat@kivy.org>

Little example to demonstrate how to start an Intent, and get the result.
When you use the Android.startActivityForResult(), the result will be
dispatched into onActivityResult. You can catch the event with the
android.activity API from python-for-android project.

If you want to compile it, don't forget to add the CAMERA permission::

    ./build.py --name 'TakePicture' --package org.test.takepicture \
            --permission CAMERA --version 1 \
            --private ~/code/kivy/examples/android/takepicture \
            debug installd

'''

__version__ = '0.1'

from kivy.app import App
from os.path import exists
from jnius import autoclass, cast
from android import activity, mActivity
from functools import partial
from kivy.clock import Clock
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import StringProperty
from kivy.network.urlrequest import UrlRequest

from PIL import Image
import base64
import urllib
import json

Intent = autoclass('android.content.Intent')
MediaStore = autoclass('android.provider.MediaStore')
Uri = autoclass('android.net.Uri')
Environment = autoclass('android.os.Environment')


class Picture(Scatter):
    source = StringProperty(None)

class ResultText(Scatter):
    text = StringProperty('')


class TakePictureApp(App):
    camsource = StringProperty(None)
    pred1source = StringProperty(None)
    pred1text = StringProperty('')
    pred2source = StringProperty(None)
    pred2text = StringProperty('')
    pred3source = StringProperty(None)
    pred3text = StringProperty('')

    def build(self):
        self.index = 0
        activity.bind(on_activity_result=self.on_activity_result)

    def get_filename(self):
        while True:
            self.index += 1
            fn = (Environment.getExternalStorageDirectory().getPath() +
                  '/takepicture{}.jpg'.format(self.index))
            if not exists(fn):
                return fn

    def take_picture(self):
        intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        self.last_fn = self.get_filename()
        self.uri = Uri.parse('file://' + self.last_fn)
        self.uri = cast('android.os.Parcelable', self.uri)
        intent.putExtra(MediaStore.EXTRA_OUTPUT, self.uri)
        mActivity.startActivityForResult(intent, 0x123)

    def on_activity_result(self, requestCode, resultCode, intent):
        if requestCode == 0x123:
            Clock.schedule_once(partial(self.add_picture, self.last_fn), 0)

    def add_picture(self, fn, *args):
        im = Image.open(fn)

        #scale image to reduce network traffic
        width, height = im.size

        if width > height:
            size = width * 250 / height
        else:
            size = height * 250 / width
        im.thumbnail((size, size), Image.ANTIALIAS)
        im.save(fn, quality=95)
        with open(fn, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data)

        body = json.dumps({'file': encoded})
        headers = {'Content-type': 'application/json',
                   'Accept': 'application/json'}
        print('python sending request')

        req = UrlRequest('http://ip5wke.hopto.org:8888/',
                                    on_success=self._inference_response,
                             on_error=self._inference_error,
                             on_failure=self._inference_failure,
                             req_body=body,
                         req_headers=headers)

        print('python sent request')
        self.camsource = fn

    def add_result(self, result, *args):
        self.pred1source = 'classimages/' + result['classes'][0] + '.PNG'
        self.pred2source = 'classimages/' + result['classes'][1] + '.PNG'
        self.pred3source = 'classimages/' + result['classes'][2] + '.PNG'
        self.pred1text = 'class: ' + result['classes'][0] + ', ' \
                                                            'prob: {' \
                                                            '0:.1f}%'.format(
            result['scores'][0] * 100)
        self.pred2text = 'class: ' + result['classes'][1] + ', prob: {0:.1f}%'.format(
            result['scores'][1] * 100)
        self.pred3text = 'class: ' + result['classes'][2] + ', prob: {0:.1f}%'.format(
            result['scores'][2] * 100)
        #self.root.canvas.ask_update()

    def _inference_response(self, req, result):
        text = result['result']

        print('python ' + str(text))
        Clock.schedule_once(partial(self.add_result, text), 0)

    def _inference_error(self, req, error):
        Clock.schedule_once(partial(self.show_error, 'Request Failure'), 0)

    def _inference_failure(self, req, error):
        Clock.schedule_once(partial(self.show_error, 'Request Failure'), 0)

    def show_error(self, text, *args):
        popup = Popup(title='Error',
            content=Label(text=text),
            size_hint=(None, None), size=(400, 400))
        popup.open()

    def on_pause(self):
        return True

    def on_resume(self):
        # after close the camera, we need to resume our app.
        print('python resumed')


TakePictureApp().run()
