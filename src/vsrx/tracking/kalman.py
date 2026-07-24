from __future__ import annotations

import numpy as np


class BBoxKalman:
    """Constant-velocity Kalman filter over cx, cy, width and height."""

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        self.state = np.array([cx, cy, width, height, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64) * 10.0
        self.transition = np.eye(8, dtype=np.float64)
        for index in range(4):
            self.transition[index, index + 4] = 1.0
        self.observation = np.zeros((4, 8), dtype=np.float64)
        self.observation[:4, :4] = np.eye(4)
        self.process_noise = np.eye(8, dtype=np.float64) * 0.05
        self.measurement_noise = np.eye(4, dtype=np.float64) * 4.0

    @staticmethod
    def _measurement(bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)],
            dtype=np.float64,
        )

    def predict(self, steps: int = 1) -> tuple[int, int, int, int]:
        for _ in range(max(1, steps)):
            self.state = self.transition @ self.state
            self.covariance = (
                self.transition @ self.covariance @ self.transition.T + self.process_noise
            )
        return self.bbox

    def update(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        measurement = self._measurement(bbox)
        innovation = measurement - self.observation @ self.state
        covariance = (
            self.observation @ self.covariance @ self.observation.T + self.measurement_noise
        )
        gain = self.covariance @ self.observation.T @ np.linalg.inv(covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(8) - gain @ self.observation) @ self.covariance
        return self.bbox

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        cx, cy, width, height = self.state[:4]
        return (
            int(round(cx - width / 2)),
            int(round(cy - height / 2)),
            int(round(cx + width / 2)),
            int(round(cy + height / 2)),
        )
