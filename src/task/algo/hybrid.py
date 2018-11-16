import numpy as np
from src.task import correlation
import pandas as pd

def calcHybridScores(pearsonScores, personalityScores, alpha = 0.4):
	return alpha * pearsonScores + (1 - alpha) * personalityScores