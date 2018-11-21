import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

questions = [
	"I'm the kind of person who prefers to be reserved.",
	"I'm the kind of person who is generally trusting.",
	"I'm a lazy kind of person.",
	"I'm usually relaxed and handles stress well.",
	"I have a few interests in artistic fields.",
	"I'm a social and outgoing person.",
	"I usually tend to find fault in others.",
	"I'm the kind of person who does a job thoroughly.",
	"I get nervous easily.",
	"I have an active imagination."
]

optionLabel = [
	"Strongly Disagree",
	"Disagree a Little",
	"Neutral",
	"Agree a Little",
	"Strongly Agree"
]


class MyApp(QWidget):
	window_width = 1200
	window_height = 600
	
	def __init__(self):
		super(MyApp, self).__init__()
		self.scores = None
		self.optionsCollection = pd.DataFrame(columns = [*[i + 1 for i in range(len(questions))]],
		                                      index = [*[i + 1 for i in range(len(optionLabel))]])
		self.setFixedSize(MyApp.window_width, MyApp.window_height)
		self.setWindowTitle("Alpha")
		self.initUI()
	
	def createLayout_QuestionGroup(self, index, question):
		groupBox = QGroupBox("Question {}:".format(index), self)
		groupLayout = QVBoxLayout(groupBox)
		
		questionLabel = QLabel(question, self)
		questionLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		questionLabel.setAlignment(Qt.AlignLeft)
		
		groupLayout.addWidget(questionLabel)
		
		options = QHBoxLayout()
		
		for oIndex, option in enumerate(optionLabel, 1):
			item = QRadioButton(option, groupBox)
			self.optionsCollection.loc[oIndex, index] = item
			if oIndex == 3:
				item.setChecked(True)
			options.addWidget(item)
		
		groupLayout.addLayout(options)
		groupLayout.addStretch(1)
		return groupBox
	
	def createLayout_Container(self):
		self.scrollarea = QScrollArea(self)
		self.scrollarea.setFixedWidth(MyApp.window_width - 20)
		self.scrollarea.setWidgetResizable(True)
		
		widget = QWidget()
		self.scrollarea.setWidget(widget)
		self.layout_SArea = QVBoxLayout(widget)
		
		for index, question in enumerate(questions, 1):
			self.layout_SArea.addWidget(self.createLayout_QuestionGroup(index, question))
		
		submitButton = QPushButton('Submit')
		submitButton.setToolTip('Submit Answer')
		submitButton.clicked.connect(self.onSubmit)
		self.layout_SArea.addWidget(submitButton)
		
		self.layout_SArea.addStretch(1)
	
	def onSubmit(self):
		answers = []
		for i in self.optionsCollection.columns:
			found = False
			for j in self.optionsCollection.index:
				if self.optionsCollection.loc[j, i].isChecked():
					found = True
					answers.append(j)
					break
			if found is False:
				answers.append(3)
		self.scores = answers
		self.close()
	
	def initUI(self):
		self.createLayout_Container()
		self.layout_All = QVBoxLayout(self)
		self.layout_All.addWidget(self.scrollarea)
