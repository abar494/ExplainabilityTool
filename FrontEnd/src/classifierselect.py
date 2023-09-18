from PyQt5.QtWidgets import QWidget, QComboBox, QLineEdit, QLabel, QPushButton, \
      QSpacerItem, QHBoxLayout, QSizePolicy, QRadioButton, QVBoxLayout
import main
import gallery
import dataprocessor
from PyQt5.QtGui import QFont
import json
  

class ClassifierSelect(QWidget):

    def __init__(self, stack, modelData=None):
        super().__init__()
        f = open('FrontEnd/src/ClassifierPrefs.json')
        self.params = {}
        self.stack = stack
        self.hps = {}
        data = json.load(f)
        self.modelData = modelData
        print(self.modelData)
        #Add title
        self.title = QLabel(self)
        self.title.setGeometry(190, 30, 391, 61)
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.title.setText("Build a Model")


        self.step1Space = QWidget(self)
        self.step1Space.setGeometry(40, 120, 201, 261)
        self.step1Vbox = QVBoxLayout(self.step1Space)
        self.step1Vbox.setContentsMargins(0, 0, 0, 0)
        self.step1Lab = QLabel(self.step1Space)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.step1Lab.setFont(font)
        self.step1Lab.setText("Step 1: Select your classifier:")
        self.step1Vbox.addWidget(self.step1Lab)

        #Add menu to choose classifier
        for clf in data.keys():
            self.clfHbox = QHBoxLayout()
            self.clfLab = QLabel(clf, self.step1Space)
            self.clfHbox.addWidget(self.clfLab)
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.clfHbox.addItem(spacerItem)
            self.clfRb = QRadioButton(self.step1Space)
            self.clfRb.setText("")
            self.clfRb.setObjectName(clf)
            self.clfRb.clicked.connect(lambda: self.clfSelect(data))
            self.clfHbox.addWidget(self.clfRb)
            self.step1Vbox.addLayout(self.clfHbox)

        # Step 2 Vertical Box Layout
        self.step2Space = QWidget(self)
        self.step2Space.setGeometry(280, 150, 422, 201)
        self.step2Vbox = QVBoxLayout(self.step2Space)
        self.step2Vbox.setContentsMargins(0, 0, 0, 0)

        #Add step 2 label to vbox
        hbox = QHBoxLayout()
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hbox.addItem(spacerItem)
        self.step2Lab = QLabel(self.step2Space)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.step2Lab.setFont(font)
        self.step2Lab.setText("Step 2: Select your hyperparameters:")
        hbox.addWidget(self.step2Lab)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hbox.addItem(spacerItem)
        self.step2Vbox.addLayout(hbox)


        # Add hyperparameter selection hbox to vbox
        self.hpHbox = QHBoxLayout()
        self.hpHbox.setObjectName("hpHbox")

        #Add three hyperparameter options
        for i in range(1, 4):
            self.hpVbox = QVBoxLayout()
            self.hpVbox.setObjectName("hp" + str(i) + "Vbox")


            hbox = QHBoxLayout()
            lb = QLabel("Param:", self.step2Space)
            hbox.addWidget(lb)
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            hbox.addItem(spacerItem)
            self.hpLab = QLabel(self.step2Space)
            self.hpLab.setObjectName("hp" + str(i) + "Lab")
            self.hpLab.setText("hp" + str(i))
            hbox.addWidget(self.hpLab)

            self.hpVbox.addLayout(hbox)


            hbox = QHBoxLayout()
            lb = QLabel("Value:", self.step2Space)
            hbox.addWidget(lb)
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            hbox.addItem(spacerItem)
            self.hpComb = QComboBox(self.step2Space)
            self.hpComb.setObjectName("hp" + str(i) + "Comb")
            self.hpComb.last = "default"
            self.hpComb.currentTextChanged.connect(self.comboChanged)
            hbox.addWidget(self.hpComb)
            self.hpVbox.addLayout(hbox)


            hbox = QHBoxLayout()
            lb = QLabel("Custom:", self.step2Space)
            hbox.addWidget(lb)
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            hbox.addItem(spacerItem)
            self.hpCustLab = QLabel("N/A", self.step2Space)
            self.hpCustLab.setObjectName("hp" + str(i) + "CustLab")
            hbox.addWidget(self.hpCustLab)
            self.hpCust = QLineEdit(self.step2Space)
            self.hpCust.setObjectName("hp" + str(i) + "Cust")
            hbox.addWidget(self.hpCust)
            self.hpVbox.addLayout(hbox)

            self.hpCustLab.show()
            self.hpCust.hide()
            self.hpHbox.addLayout(self.hpVbox)

        # Add hyperparameter selection hbox to vbox
        self.step2Vbox.addLayout(self.hpHbox)

        #Initialise variable hyperparameter selection
        self.VarHpHbox = QHBoxLayout()
        self.VarHpLab = QLabel(self.step2Space)
        self.VarHpLab.setText("Variable Hyperparameter")
        self.VarHpComb = QComboBox(self.step2Space)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        #Add variable hyperparameter selection to to Step 2 vbox
        self.VarHpHbox.addWidget(self.VarHpLab)
        self.VarHpHbox.addWidget(self.VarHpComb)
        self.VarHpHbox.addItem(spacerItem)
        self.step2Vbox.addLayout(self.VarHpHbox)

        #Add accept button
        self.accept = QPushButton("Accept", self)
        self.accept.setGeometry(580, 400, 132, 32)
        self.accept.clicked.connect(lambda: self.modelSelected())
        self.accept.hide()

        #Add back button
        back = QPushButton("Back", self)
        back.clicked.connect(lambda: main.transition(stack, main.MainMenu(stack)))
        self.show()
    

    def clfSelect(self, data):
        self.clf = self.sender().objectName()
        self.call = data[self.clf]["call"]
        self.hps = data[self.clf]["hps"]
        self.VarHpComb.clear()

        i = 1
        for hp in self.hps.keys():
            self.VarHpComb.addItem(hp)
            obj = self.findChild(QLabel, "hp"+str(i)+"Lab")
            obj.setText(hp)
            com = self.findChild(QComboBox, "hp"+str(i)+"Comb")
            com.clear()
            com.addItem("Default")
            for val in self.hps[hp]:
                com.addItem(val)
            i+=1
        self.accept.show()
        self.show()

    
    def comboChanged(self):
        box = self.sender()
        combName = box.objectName()
        combNum = ''.join(filter(str.isdigit, combName))
        obj = self.findChild(QLabel, "hp"+str(combNum)+"Lab")
        hpName = obj.text()
        hpVal = box.currentText()
        print(hpVal)
        self.params[hpName] = hpVal
        notApplic = self.findChild(QLabel, "hp"+str(combNum)+"CustLab")
        custom = self.findChild(QLineEdit, "hp"+str(combNum)+"Cust")
        # if hpVal == "Custom" and (custom is None):
        #     p = box.pos() + QPoint(80, 120)
        #     line = QLineEdit(self)
        #     line.setPlaceholderText("Enter Value")
        #     line.setObjectName(hpName)
        #     line.move(p)
        #     line.show()
        if hpVal == "Custom":
            custom.show()
            notApplic.hide()
        else:
            if box.last == "Custom":
                custom.setText("")
                custom.hide()
                notApplic.show()
            if hpVal == "Default":
                self.params.pop(hpName)
        box.last = hpVal


    def modelSelected(self):
        for hpName in self.params.keys():
            if self.params[hpName].isdigit():
                self.params[hpName] = int(self.params[hpName])
            if self.params[hpName] == "Custom":
                for i in range(1, len(self.hps.keys())+1):
                    lab = self.findChild(QLabel, "hp"+str(i)+"Lab")
                    if lab.text() == hpName:
                        obj = self.findChild(QLineEdit, "hp"+str(i)+"Cust")
                        if obj.text() == "":
                            self.params.pop(hpName)
                        else:
                            self.params[hpName] = float(obj.text())
                        break
        vals = self.hps[self.VarHpComb.currentText()]
        if "Custom" in vals:
            vals.remove("Custom")
            vals = [int(val) for val in vals]
        prefs = {
            "hps": self.params,
            "clf": self.call,
            "var": self.VarHpComb.currentText(),
            "vals": vals
        }

        dp = dataprocessor.DataProcessor(prefs, self.modelData)
        self.modelData = {
            "x_test": dp.x_test,
            "y_test": dp.y_test,
            "iclf": dp.iclf,
            "clfs": dp.clfs,
            "vals": dp.vals,
            "var": dp.var
        }
        main.transition(self.stack, gallery.Gallery(self.stack, self.modelData))