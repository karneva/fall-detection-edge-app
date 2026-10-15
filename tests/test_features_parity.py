"""
features.v1이 기존에 세 곳에 복제돼 있던 정규화 구현과 비트 단위로 같은지 고정합니다.
(아래 _old_* 함수는 통합 직전 코드를 그대로 옮긴 사본이며 수정하지 않습니다.)
"""
import importlib.util
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _old_normalize_keypoints(kps):
    # app/ai_classifier/dataset_builder.py, video_to_data.py (통합 전)
    import numpy as np
    l_hip = kps[11]
    r_hip = kps[12]
    if np.any(l_hip) and np.any(r_hip):
        origin = (l_hip + r_hip) / 2
    elif np.any(l_hip):
        origin = l_hip
    elif np.any(r_hip):
        origin = r_hip
    else:
        origin = kps[0] if np.any(kps[0]) else np.array([0.5, 0.5])
    norm_kps = kps - origin
    return norm_kps.flatten()


def _old_normalize_sequence(sequence):
    # app/ai_classifier/inference.py FallClassifier.normalize_sequence (통합 전)
    import numpy as np
    norm_seq = []
    for frame in sequence:
        kps = frame.reshape(17, 2)
        l_hip = kps[11]
        r_hip = kps[12]
        if np.any(l_hip) and np.any(r_hip):
            origin = (l_hip + r_hip) / 2
        elif np.any(l_hip):
            origin = l_hip
        elif np.any(r_hip):
            origin = r_hip
        else:
            origin = kps[0] if np.any(kps[0]) else np.array([0.5, 0.5])
        norm_frame = (kps - origin).flatten()
        norm_seq.append(norm_frame)
    return np.array(norm_seq, dtype=np.float32)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class FeaturesV1ParityTest(unittest.TestCase):
    def cases(self):
        import numpy as np
        rng = np.random.default_rng(0)
        base = rng.random((17, 2)).astype(np.float32)
        out = {"random": base}
        both_zero = base.copy(); both_zero[11] = 0; both_zero[12] = 0
        out["no_hips"] = both_zero
        left_only = base.copy(); left_only[12] = 0
        out["left_hip_only"] = left_only
        right_only = base.copy(); right_only[11] = 0
        out["right_hip_only"] = right_only
        nothing = both_zero.copy(); nothing[0] = 0
        out["no_hips_no_nose"] = nothing
        out["all_zero"] = np.zeros((17, 2), dtype=np.float32)
        out["float64"] = rng.random((17, 2))
        return out

    def test_frame_parity(self):
        import numpy as np
        from app.ai_classifier.features import normalize_frame_v1

        for name, kps in self.cases().items():
            old, new = _old_normalize_keypoints(kps), normalize_frame_v1(kps)
            self.assertEqual(old.dtype, new.dtype, name)
            self.assertTrue(np.array_equal(old, new), name)

    def test_sequence_parity(self):
        import numpy as np
        from app.ai_classifier.features import normalize_sequence_v1

        frames = [k.astype(np.float32).reshape(34) for k in self.cases().values()]
        rng = np.random.default_rng(1)
        frames += list(rng.random((30, 34)).astype(np.float32))
        seq = np.array(frames, dtype=np.float32)
        old, new = _old_normalize_sequence(seq), normalize_sequence_v1(seq)
        self.assertEqual(new.dtype, np.float32)
        self.assertTrue(np.array_equal(old, new))

    def test_classifier_uses_shared_implementation(self):
        import numpy as np
        from app.ai_classifier.inference import _DisabledClassifier

        seq = np.random.default_rng(2).random((30, 34)).astype(np.float32)
        clf = _DisabledClassifier("test")
        self.assertTrue(np.array_equal(clf.normalize_sequence(seq), _old_normalize_sequence(seq)))


if __name__ == "__main__":
    unittest.main()
