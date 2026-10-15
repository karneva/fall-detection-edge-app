"""
운영 모드 기본 인터페이스
QR 모드와 Live 모드 등 모든 실행 모드가 상속받아야 하는 기본 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any


class BaseMode(ABC):
    """
    애플리케이션 내의 특정 실행 로직(모드)을 정의하는 추상 클래스입니다.
    """
    
    def __init__(self, name: str):
        """
        모드를 초기화합니다.
        
        Args:
            name: 모드의 식별 이름
        """
        self.name = name
        self.is_running = False
    
    @abstractmethod
    def setup(self) -> bool:
        """
        모드 구동에 필요한 초기화(엔진 로드, 하위 객체 생성 등)를 수행합니다.
        성공 시 True를 반환합니다.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        모드 종료 시 점유하고 있던 리소스를 해제하거나 상태를 정리합니다.
        """
        pass
    
    @abstractmethod
    def step(self) -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[Any], Optional[Any]]:
        """
        한 프레임에 대한 처리 로직을 실행합니다.
        
        Returns:
            (관측결과, 시각화데이터, 원본프레임, 추가데이터) 튜플을 반환합니다.
        """
        pass
