"""
기기 인증 및 보안 관리 모듈
QR 코드 페어링 토큰 생성, 등록 토큰 해싱 및 검증 루틴을 담당합니다.
"""

import secrets
import hashlib
import time
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DeviceAuth:
    """
    장치 보안 인증을 위한 유틸리티 클래스입니다.
    """
    
    TOKEN_EXPIRY = 180  # 토큰 유효 기간 (3분)
    TOKEN_LENGTH = 32   # 생성할 토큰의 비트 길이
    
    @staticmethod
    def generate_qr_token() -> str:
        """QR 코드에 담길 일회용 페어링 토큰을 생성합니다."""
        return secrets.token_hex(DeviceAuth.TOKEN_LENGTH // 2)
    
    @staticmethod
    def generate_registration_token() -> str:
        """장치 등록 시 사용할 고유 토큰을 생성합니다."""
        return secrets.token_hex(32)
    
    @staticmethod
    def hash_token(token: str) -> str:
        """보안을 위해 토큰 문자열을 SHA-256으로 해싱합니다."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    @staticmethod
    def verify_token(provided_token: str, stored_hash: str) -> bool:
        """
        제공된 토큰과 저장된 해시값이 일치하는지 검증합니다.
        secrets.compare_digest를 사용하여 타이밍 공격(Timing Attack)을 방지합니다.
        """
        provided_hash = DeviceAuth.hash_token(provided_token)
        return secrets.compare_digest(provided_hash, stored_hash)
    
    @staticmethod
    def is_token_valid(token_created_at: float) -> bool:
        """토큰이 만료 시간 이내에 생성되었는지 확인합니다."""
        return (time.time() - token_created_at) < DeviceAuth.TOKEN_EXPIRY


# 메모리 기반 임시 토큰 저장소 (실제 운영 시 Redis 등을 통해 지속성 보장 권장)
_qr_token_store = {}  # token -> {created_at, device_id, verified}


def create_qr_token() -> Tuple[str, float]:
    """
    새로운 QR 페어링 토큰을 생성하고 저장소에 등록합니다.
    """
    token = DeviceAuth.generate_qr_token()
    created_at = time.time()
    _qr_token_store[token] = {
        "created_at": created_at,
        "device_id": None,
        "verified": False
    }
    logger.info(f"New QR pairing token created (valid for 3m)")
    return token, created_at


def verify_qr_token(token: str) -> bool:
    """요청된 QR 토큰의 존재 여부와 만료 시간을 검증합니다."""
    if token not in _qr_token_store:
        logger.warning(f"QR token not found or already used: {token[:8]}...")
        return False
    
    token_data = _qr_token_store[token]
    if not DeviceAuth.is_token_valid(token_data["created_at"]):
        logger.warning("QR token has expired")
        del _qr_token_store[token]
        return False
    
    return True


def get_qr_token(token: str) -> Optional[dict]:
    """특정 QR 토큰에 담긴 상세 정보를 조회합니다."""
    return _qr_token_store.get(token)


def complete_qr_token(token: str, device_id: str) -> bool:
    """페어링이 성공적으로 완료되었음을 표시하고 장치 ID를 매핑합니다."""
    if token not in _qr_token_store:
        return False
    
    _qr_token_store[token]["device_id"] = device_id
    _qr_token_store[token]["verified"] = True
    logger.info(f"QR pairing completed for device: {device_id}")
    return True


def cleanup_expired_tokens():
    """만료된 모든 토큰을 저장소에서 일괄 정리합니다."""
    current_time = time.time()
    expired = [
        token for token, data in _qr_token_store.items()
        if (current_time - data["created_at"]) > DeviceAuth.TOKEN_EXPIRY
    ]
    for token in expired:
        del _qr_token_store[token]
    
    if expired:
        logger.info(f"Purged {len(expired)} expired security tokens")
