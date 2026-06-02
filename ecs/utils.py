"""clock.py - raylib-based clock allowing to 'drain' it such as to keep a steady FPS rate and consitent physics"""
import raylib as rl

EntityId = int
Shape = tuple[int, ...]

class Clock:
    """clock used for physics with fixed DT in main loops"""

    def __init__(self, dt: float, max_ticks: int):
        self.dt = dt
        self.max_ticks = max_ticks
        self.prev_time = rl.GetTime()
        self.accumulator = 0

    def tick(self):
        """tick once by adding the delta between prev frame and now"""
        now = rl.GetTime()
        frame_time = now - self.prev_time
        self.prev_time = now
        self.accumulator += frame_time

    def drain(self):
        """drain the accumulator. in main loop: for _ in clock.drain(): ..."""
        n_ticks = 0
        while self.accumulator >= self.dt and n_ticks < self.max_ticks:
            yield
            self.accumulator -= self.dt
            n_ticks += 1
        self.accumulator = min(self.accumulator, self.dt) # Drop residual debt instead of it piling up across frames

    def wait(self):
        """waits the leftover time in case the previous tick ran too fast to maintain consistent FPS"""
        rl.WaitTime(max(self.dt - (rl.GetTime() - self.prev_time), 0))
