import time


class Timing:
	startTime = None
	
	def __init__(self) -> None:
		super().__init__()
		self.reset()
	
	def reset(self):
		self.startTime = time.time()
	
	def getElapsedTime(self):
		return time.time() - self.startTime
	
	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		pass
