# !/usr/bin/python3
# -*- coding: utf-8 -*-
SETTINGS_PRIORITY = 0

import os

def path(filename):
    return os.path.join(os.path.dirname(__file__), 'resources', filename)

ANNOTATOR_ICON_IDTRACKERAI = path('idtrackerai.png')

IDTRACKERAI_SHORT_KEYS= {
    'Go to next crossing.':                     'Ctrl+S',
    'Go to previous crossing.':                 'Ctrl+A',
    'Go to next crossing non strict.':          'Ctrl+W',
    'Go to previous crossing non strict.':      'Ctrl+Q',
    'Go to next ok frame.':                     'Alt+S',
    'Go to previous ok frame.':                 'Alt+A',
    'Check/Uncheck add centroid.':              'Ctrl+C',
    'Check/Uncheck add blob.':                  'Ctrl+B',
    'Delete centroid.':                         'Ctrl+D'
}