import torch
import numpy as np
import os
from app.ai_classifier.model import Fall1DCNN

class FallClassifier:
    def __init__(self, model_path="app/ai_classifier/weights/fall_model_active_best.pth", threshold=0.6, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = Fall1DCNN().to(self.device)
        self.threshold = threshold
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded AI classifier model from {model_path} (threshold={threshold})")
        else:
            print(f"Warning: Model weights not found at {model_path}. Using uninitialized model.")
            
        self.model.eval()
        self.seq_len = 30
        
    def normalize_sequence(self, sequence):
        """
        시퀀스의 각 프레임에 대해 골반 중심 정규화 적용.
        sequence: Array of (30, 34)
        """
        norm_seq = []
        for frame in sequence:
            kps = frame.reshape(17, 2)
            # 11: Left Hip, 12: Right Hip
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

    def predict(self, keypoint_sequence):
        """
        keypoint_sequence: List of 30 frames, each frame is 34-dim array
        Returns:
            label: 1 for Fall, 0 for ADL, None if uncertain
            confidence: float
        """
        if len(keypoint_sequence) < self.seq_len:
            return None, 0.0
            
        # 정규화 적용
        seq_np = np.array(keypoint_sequence, dtype=np.float32)
        seq_np = self.normalize_sequence(seq_np) # (30, 34)
        
        # [30, 34] -> [1, 34, 30] (Batch, Channel, Length)
        seq_np = np.transpose(seq_np, (1, 0))
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(seq_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        conf_val = confidence.item()
        label_val = predicted.item()
        
        # 신뢰도 임계값 체크
        if conf_val < self.threshold:
            return None, conf_val # 불확실하면 판단 보류 (Rule-based 따르기)
            
        return label_val, conf_val

# Singleton instance
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = FallClassifier()
    return _classifier
