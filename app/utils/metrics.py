"""
성능 메트릭 수집 & 분석 유틸

사용법:
  from .metrics import MetricsCollector
  
  metrics = MetricsCollector("runs/metrics.jsonl")
  
  # 이벤트 기록
  metrics.record_event("abnormal_enter", level1_engine.get_confidence())
  metrics.record_event("level1")
  metrics.record_fps(current_fps)
  
  # 주기적 리포트
  metrics.print_summary(interval=100)
"""

import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any


class MetricsCollector:
    """
    실시간 메트릭 수집 및 분석
    """
    
    def __init__(self, log_path: str = "runs/metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 저장소
        self.counters = defaultdict(int)
        self.timings = defaultdict(list)
        self.start_time = time.time()
        self.frame_count = 0
        
        # 시간 추적 (abnormal_enter → level1 레이턴시 측정)
        self.pending_events = {}  # event_id -> timestamp
        
    def record_event(self, event_type: str, metadata: Dict[str, Any] = None):
        """
        이벤트 기록
        
        인수:
            event_type: 이벤트 타입 (abnormal_enter, abnormal_exit, level1)
            metadata: 추가 메타데이터 (confidence, keypoint_count 등)
        """
        self.counters[event_type] += 1
        
        event_data = {
            "ts": time.time(),
            "type": event_type,
            "total_events": sum(self.counters.values()),
        }
        
        if metadata:
            event_data.update(metadata)
        
        # 레이턴시 추적
        if event_type == "abnormal_enter":
            event_id = self.counters[event_type]
            self.pending_events[event_id] = time.time()
            event_data["event_id"] = event_id
            
        elif event_type == "level1":
            # 가장 최근 pending 이벤트와 매칭
            if self.pending_events:
                event_id = max(self.pending_events.keys())
                start_ts = self.pending_events.pop(event_id)
                latency = time.time() - start_ts
                self.timings["abnormal_to_level1"].append(latency)
                event_data["latency_ms"] = latency * 1000
        
        # JSONL로 저장
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
    
    def record_fps(self, fps: float):
        """FPS 기록"""
        self.timings["fps"].append(fps)
    
    def get_summary(self) -> Dict[str, Any]:
        """현재 메트릭 요약"""
        elapsed = time.time() - self.start_time
        
        summary = {
            "elapsed_time_sec": round(elapsed, 1),
            "total_events": sum(self.counters.values()),
            "event_breakdown": dict(self.counters),
        }
        
        # 레이턴시 통계
        if self.timings["abnormal_to_level1"]:
            latencies = self.timings["abnormal_to_level1"]
            summary["latency_stats"] = {
                "avg_ms": round(sum(latencies) / len(latencies) * 1000, 1),
                "min_ms": round(min(latencies) * 1000, 1),
                "max_ms": round(max(latencies) * 1000, 1),
                "samples": len(latencies),
            }
        
        # FPS 통계
        if self.timings["fps"]:
            fps_list = self.timings["fps"]
            summary["fps_stats"] = {
                "avg": round(sum(fps_list) / len(fps_list), 1),
                "min": round(min(fps_list), 1),
                "max": round(max(fps_list), 1),
            }
        
        # 정확도 계산 (level1 / abnormal_enter)
        if self.counters["abnormal_enter"] > 0:
            accuracy = self.counters["level1"] / self.counters["abnormal_enter"]
            summary["accuracy"] = {
                "level1_rate": round(accuracy * 100, 1),
                "false_positive_rate": round((1 - accuracy) * 100, 1),
            }
        
        # False positive 추정 (abnormal_enter - level1 확정 안 된 것)
        unconfirmed = len(self.pending_events)
        if unconfirmed > 0:
            summary["pending_events"] = unconfirmed
        
        return summary
    
    def print_summary(self, interval: int = 100, force: bool = False):
        """
        주기적 요약 출력
        
        인수:
            interval: 이 주기마다 출력 (기본 100 이벤트)
            force: 강제 출력 여부
        """
        total = sum(self.counters.values())
        
        if not force and (total == 0 or total % interval != 0):
            return
        
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"[METRICS] 요약 - {summary['elapsed_time_sec']}초 경과")
        print("="*60)
        
        # 이벤트 분포
        print(f"전체 이벤트: {summary['total_events']}")
        for event_type, count in summary['event_breakdown'].items():
            print(f"  - {event_type}: {count}")
        
        # 정확도
        if "accuracy" in summary:
            acc = summary["accuracy"]
            print(f"\n정확도:")
            print(f"  - Level1 비율: {acc['level1_rate']}%")
            print(f"  - 거짓 양성: {acc['false_positive_rate']}%")
        
        # 레이턴시
        if "latency_stats" in summary:
            lat = summary["latency_stats"]
            print(f"\n레이턴시 (abnormal → level1):")
            print(f"  - 평균: {lat['avg_ms']}ms")
            print(f"  - 범위: {lat['min_ms']}~{lat['max_ms']}ms")
            print(f"  - 샘플: {lat['samples']}")
        
        # FPS
        if "fps_stats" in summary:
            fps = summary["fps_stats"]
            print(f"\n처리 FPS:")
            print(f"  - 평균: {fps['avg']}")
            print(f"  - 범위: {fps['min']}~{fps['max']}")
        
        # Pending
        if "pending_events" in summary:
            print(f"\n⚠️  대기 중인 이벤트: {summary['pending_events']}")
        
        print("="*60 + "\n")
    
    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        메트릭을 JSON으로 내보내기
        
        반환:
            저장된 파일 경로
        """
        if output_path is None:
            output_path = self.log_path.parent / "metrics_summary.json"
        
        summary = self.get_summary()
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 메트릭 내보내기 완료: {output_path}")
        return str(output_path)


# 사용 예제
if __name__ == "__main__":
    # 테스트
    metrics = MetricsCollector()
    
    # 시뮬레이션
    for i in range(10):
        metrics.record_event("abnormal_enter", {"confidence": 0.8})
        time.sleep(0.5)
        if i % 2 == 0:
            metrics.record_event("level1")
        else:
            metrics.record_event("abnormal_exit")
        
        metrics.record_fps(25 + (i % 5))
    
    # 요약 출력
    metrics.print_summary(force=True)
    
    # JSON 내보내기
    metrics.export_json()
