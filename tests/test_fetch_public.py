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

    def test_skips_macos_metadata(self):
        z = self.dir / "mac.zip"
        self.make_zip(z, {"Fall/a.mp4": b"v", "__MACOSX/Fall/._a.mp4": b"m", "Fall/._b.mp4": b"m"})
        dest = self.dir / "out"
        safe_extract(z, dest)
        self.assertEqual((dest / "Fall/a.mp4").read_bytes(), b"v")
        self.assertFalse((dest / "__MACOSX").exists())
        self.assertFalse((dest / "Fall/._b.mp4").exists())

    def test_non_zip_named_zip_is_left_alone(self):
        z = self.dir / "outer.zip"
        self.make_zip(z, {"notes.zip": b"plain text", "notes": b"same stem file"})
        dest = self.dir / "out"
        safe_extract(z, dest)
        self.assertEqual((dest / "notes.zip").read_bytes(), b"plain text")

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

    def test_urfd_source(self):
        names = [f[0] for f in SOURCES["urfd"]["files"]]
        self.assertEqual(len(names), 72)
        self.assertEqual(len(set(names)), 72)
        for n in ("fall-01-cam0.mp4", "fall-30-cam0.mp4", "adl-01-cam0.mp4", "adl-40-cam0.mp4",
                  "urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
            self.assertIn(n, names)
        self.assertFalse(any("cam1" in n for n in names))
        self.assertIn("NC", SOURCES["urfd"]["license"])


if __name__ == "__main__":
    unittest.main()
