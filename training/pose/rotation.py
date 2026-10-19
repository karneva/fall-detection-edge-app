"""
회전 좌표 변환 (운영 회전 폴백과 같은 구현을 다시 내보냄: app.engine.rotate_fallback)

영상을 0/90/180/270도로 돌려 검출하고 결과를 원래 좌표로 되돌릴 때 씁니다. 좌표 규칙은 app.engine.rotate_fallback 참고.
"""
from app.engine.rotate_fallback import ANGLES, box_from_rotated, from_rotated, rotated_size, to_rotated

__all__ = ["ANGLES", "rotated_size", "to_rotated", "from_rotated", "box_from_rotated"]
