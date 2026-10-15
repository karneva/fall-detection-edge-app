import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from training.datasets.fetch_public import SOURCES, main, safe_extract


class SafeExtractTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_zip(self, path, entries):
        with zipfile.ZipFile(path, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)

    def test_extracts_nested_zip(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("clip/video.avi", b"v")
        outer = self.dir / "outer.zip"
        self.make_zip(outer, {"a/readme.txt": b"r", "a/scene.zip": inner.getvalue()})
        dest = self.dir / "out"
        safe_extract(outer, dest)
        self.assertEqual((dest / "a/readme.txt").read_bytes(), b"r")
        self.assertEqual((dest / "a/scene/clip/video.avi").read_bytes(), b"v")
        self.assertFalse((dest / "a/scene.zip").exists())

    def test_rejects_path_traversal(self):
        bad = self.dir / "bad.zip"
        self.make_zip(bad, {"../escape.txt": b"x"})
        with self.assertRaises(ValueError):
            safe_extract(bad, self.dir / "out")
        self.assertFalse((self.dir / "escape.txt").exists())

    def test_rejects_tar_symlink(self):
        bad = self.dir / "bad.tar"
        with tarfile.open(bad, "w") as t:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            t.addfile(info)
        with self.assertRaises(ValueError):
            safe_extract(bad, self.dir / "out")

    def test_sources_registry(self):
        for name, src in SOURCES.items():
            self.assertTrue(src["license"] and src["page"] and src["files"], name)
            for fname, url, algo, expected in src["files"]:
                self.assertTrue(url.startswith("https://"), (name, fname))
                self.assertEqual(algo is None, expected is None, (name, fname))
        self.assertEqual(main(["list"]), 0)


if __name__ == "__main__":
    unittest.main()
