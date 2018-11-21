import sys

from PyQt5.QtWidgets import *

from src.task.guiScores import MyApp


def getScores():
	app = QApplication(sys.argv)
	window = MyApp()
	window.show()
	app.exec_()
	
	answers = window.scores
	
	if answers is None:
		print("Invalid Input")
		sys.exit(-1)
	
	return answers


def calcPScores(score):
	e = ((6 - score[0] + score[5]) / 2) * 7 / 5
	a = ((6 - score[6] + score[1]) / 2) * 7 / 5
	c = ((6 - score[2] + score[7]) / 2) * 7 / 5
	n = ((6 - score[3] + score[8]) / 2) * 7 / 5
	o = ((6 - score[4] + score[9]) / 2) * 7 / 5
	
	return (o, c, e, a, n)


def run():
	userPScore = calcPScores(getScores())
	
	import src.app.iMode as app
	app.run(True, userPScore)
