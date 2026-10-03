"""Host checks for explicit feedback admission and serialized local operations."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from live_ui.studio import text
from live_ui.native import quoted

class AdmissionTests(unittest.TestCase):
    def test_reject_control_and_empty(self):
        for value in ('', '  ', 'x\ny', 'x\ry', 'x\x1ey', 'x\0y', None, 14):
            with self.subTest(value=value), self.assertRaises(ValueError):
                text(value,1024)
    def test_utf8_byte_limit(self):
        self.assertEqual(text(' é ', 4),'é')
        with self.assertRaises(ValueError): text('ééé',4)
    def test_protocol_escaping(self):
        self.assertEqual(quoted('a"b\\c'), '"a\\"b\\\\c"')
    def test_markup_is_plain_text(self):
        self.assertEqual(text('<script>alert(1)</script>',100),'<script>alert(1)</script>')

if __name__=='__main__': unittest.main()
