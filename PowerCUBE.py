import sys
import time
import math
import random
import json
import numpy as np
from numba import njit, prange

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QMenuBar, QFileDialog, QSlider, QPushButton, QComboBox, QCheckBox,
                               QDockWidget, QTreeWidget, QTreeWidgetItem, QLineEdit, QTabWidget, QToolBar,
                               QMessageBox, QGraphicsOpacityEffect)
from PySide6.QtGui import QPainter, QColor, QAction, QIcon
from PySide6.QtCore import QTimer, Qt, QPoint
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import os

#--------------------------------------------
# 전문적 material db 로딩: JSON 파일로부터
#--------------------------------------------
MATERIALS_JSON = """
{
  "materials": [
    {
      "name": "Water",
      "density": 1000,
      "specificHeat": 4184,
      "thermalConductivity": 0.6,
      "electricalConductivity": 0,
      "meltingPoint": 0,
      "boilingPoint": 100,
      "color": [0.2,0.2,0.8]
    },
    {
      "name": "Ethanol",
      "density": 789,
      "specificHeat": 2440,
      "thermalConductivity": 0.17,
      "electricalConductivity": 0,
      "meltingPoint": -114,
      "boilingPoint": 78,
      "color": [0.8,0.8,0.2]
    },
    {
      "name": "NaCl",
      "density": 2160,
      "specificHeat": 850,
      "thermalConductivity": 6.5,
      "electricalConductivity": 0,
      "meltingPoint": 801,
      "boilingPoint": 1465,
      "color": [0.9,0.9,0.9]
    },
    {
      "name": "H2SO4",
      "density": 1840,
      "specificHeat": 1380,
      "thermalConductivity": 0.5,
      "electricalConductivity": 0,
      "meltingPoint": -10,
      "boilingPoint": 337,
      "color": [0.8,0.2,0.8]
    },
    {
      "name": "NaOH",
      "density": 2130,
      "specificHeat": 1400,
      "thermalConductivity": 0.2,
      "electricalConductivity": 0,
      "meltingPoint": 318,
      "boilingPoint": 1390,
      "color": [0.7,0.9,0.7]
    },
    {
      "name": "Fe",
      "density": 7874,
      "specificHeat": 449,
      "thermalConductivity": 80,
      "electricalConductivity": 10,
      "meltingPoint": 1538,
      "boilingPoint": 2862,
      "color": [0.5,0.5,0.5]
    },
    {
      "name": "Cu",
      "density": 8960,
      "specificHeat": 385,
      "thermalConductivity": 401,
      "electricalConductivity": 59,
      "meltingPoint": 1085,
      "boilingPoint": 2562,
      "color": [0.7,0.4,0.1]
    },
    {
      "name": "Au",
      "density": 19300,
      "specificHeat": 129,
      "thermalConductivity": 317,
      "electricalConductivity": 44,
      "meltingPoint": 1064,
      "boilingPoint": 2970,
      "color": [1.0,0.9,0.0]
    },
    {
      "name": "SiO2",
      "density": 2650,
      "specificHeat": 700,
      "thermalConductivity": 1.4,
      "electricalConductivity": 0,
      "meltingPoint": 1710,
      "boilingPoint": 2230,
      "color": [0.9,0.8,0.7]
    },
    {
      "name": "CO2",
      "density": 1.977,
      "specificHeat": 844,
      "thermalConductivity": 0.0146,
      "electricalConductivity": 0,
      "meltingPoint": -78.5,
      "boilingPoint": -56.6,
      "color": [0.5,0.5,0.5]
    },
    {
      "name": "O2",
      "density": 1.429,
      "specificHeat": 918,
      "thermalConductivity": 0.026,
      "electricalConductivity": 0,
      "meltingPoint": -219,
      "boilingPoint": -183,
      "color": [0.5,0.5,0.6]
    },
    {
      "name": "N2",
      "density": 1.2506,
      "specificHeat": 1040,
      "thermalConductivity": 0.026,
      "electricalConductivity": 0,
      "meltingPoint": -210,
      "boilingPoint": -196,
      "color": [0.5,0.5,0.8]
    },
    {
      "name": "HCl",
      "density": 1.639,
      "specificHeat": 1000,
      "thermalConductivity": 0.010,
      "electricalConductivity": 0,
      "meltingPoint": -114,
      "boilingPoint": -85,
      "color": [0.9,0.5,0.5]
    },
    {
      "name": "CH4",
      "density": 0.656,
      "specificHeat": 2220,
      "thermalConductivity": 0.03,
      "electricalConductivity": 0,
      "meltingPoint": -182.5,
      "boilingPoint": -161.5,
      "color": [0.8,0.7,0.9]
    }
  ]
}
"""

#--------------------------------------------
# Core Classes: Material, MaterialDatabase
#--------------------------------------------
class Material:
    def __init__(self, name, density, specificHeat, thermalConductivity, electricalConductivity,
                 meltingPoint, boilingPoint, color):
        self.name = name
        self.density = density
        self.specificHeat = specificHeat
        self.thermalConductivity = thermalConductivity
        self.electricalConductivity = electricalConductivity
        self.meltingPoint = meltingPoint
        self.boilingPoint = boilingPoint
        self.color = color

class MaterialDatabase:
    def __init__(self):
        self.materials = []
        self.nameToID = {}

    def LoadFromJSON(self, json_str):
        data = json.loads(json_str)
        self.materials.clear()
        self.nameToID.clear()
        for i, mat in enumerate(data["materials"]):
            m = Material(mat["name"], mat["density"], mat["specificHeat"], mat["thermalConductivity"],
                         mat["electricalConductivity"], mat["meltingPoint"], mat["boilingPoint"], mat["color"])
            self.materials.append(m)
            self.nameToID[m.name] = i

    def GetMaterial(self, id):
        return self.materials[id]

#--------------------------------------------
# Reaction
#--------------------------------------------
class Reaction:
    def __init__(self, r1, r2, products, A, Ea, deltaH):
        self.reactant1 = r1
        self.reactant2 = r2
        self.products = products
        self.A = A
        self.Ea = Ea
        self.deltaH = deltaH

class ReactionDatabase:
    def __init__(self):
        self.reactionMap = {}

    def AddReaction(self, r):
        for k in [r.reactant1,r.reactant2]:
            if k not in self.reactionMap:
                self.reactionMap[k] = []
            self.reactionMap[k].append(r)

    def GetReactionsFor(self, matID):
        return self.reactionMap.get(matID, [])

#--------------------------------------------
# Tools
#--------------------------------------------
class Tool:
    def __init__(self, name, x,y,w,h):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def apply(self, grid, matDB, dt, expertMode):
        pass

class Beaker(Tool):
    def apply(self, grid, matDB, dt, expertMode):
        # 벽효과, 외부 셀과 교환금지 등
        pass

class Heater(Tool):
    def apply(self, grid, matDB, dt, expertMode):
        factor = 20 if expertMode else 10
        for yy in range(self.y,self.y+self.h):
            for xx in range(self.x,self.x+self.w):
                if 0<=xx<grid.width and 0<=yy<grid.height:
                    c = grid.GetCell(xx,yy)
                    c.temperature += factor*dt

class Cooler(Tool):
    def apply(self, grid, matDB, dt, expertMode):
        factor = 10 if expertMode else 5
        for yy in range(self.y,self.y+self.h):
            for xx in range(self.x,self.x+self.w):
                if 0<=xx<grid.width and 0<=yy<grid.height:
                    c = grid.GetCell(xx,yy)
                    c.temperature -= factor*dt

#--------------------------------------------
# Cell, CellGrid
#--------------------------------------------
class Cell:
    def __init__(self, matID=0, temperature=20.0, pressure=101325.0):
        self.materialID = matID
        self.temperature = temperature
        self.pressure = pressure
        self.velocityX = 0.0
        self.velocityY = 0.0
        self.recentlyReacted = False
        self.isSpawner = False
        self.spawnMaterialID = 0

class CellGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [Cell() for _ in range(width*height)]

    def GetCell(self,x,y):
        return self.cells[y*self.width+x]

    def InBounds(self,x,y):
        return 0<=x<self.width and 0<=y<self.height

    def SetSpawner(self,x,y,matID):
        if self.InBounds(x,y):
            c = self.GetCell(x,y)
            c.isSpawner = True
            c.spawnMaterialID = matID

    def UpdateSpawners(self, spawnRate):
        for c in self.cells:
            if c.isSpawner:
                c.materialID = c.spawnMaterialID
                c.temperature = 20.0

    def AddMaterial(self,x,y,matID,brushSize=1):
        for dy in range(-brushSize,brushSize+1):
            for dx in range(-brushSize,brushSize+1):
                nx, ny = x+dx,y+dy
                if self.InBounds(nx,ny):
                    cell = self.GetCell(nx,ny)
                    cell.materialID = matID
                    cell.temperature = 20.0

#--------------------------------------------
# Engines: ReactionEngine, FluidSolver, ThermalSolver
#--------------------------------------------
class ReactionEngine:
    def __init__(self, matDB, rxDB, config):
        self.matDB = matDB
        self.rxDB = rxDB
        self.config = config
        self.R = 8.314

    def ProcessReactions(self, grid, dt):
        directions = [(1,0),(0,1)]
        if not self.config.expertMode:
            # 단순 반응
            for y in range(grid.height):
                for x in range(grid.width):
                    c1 = grid.GetCell(x,y)
                    rxList = self.rxDB.GetReactionsFor(c1.materialID)
                    for dx,dy in directions:
                        nx,ny = x+dx,y+dy
                        if grid.InBounds(nx,ny):
                            c2 = grid.GetCell(nx,ny)
                            for rx in rxList:
                                if (rx.reactant1 == c1.materialID and rx.reactant2 == c2.materialID) or \
                                   (rx.reactant2 == c1.materialID and rx.reactant1 == c2.materialID):
                                    if random.random()<0.1*dt:
                                        if rx.products:
                                            c1.materialID = rx.products[0]
                                        dH = rx.deltaH*dt
                                        c1.temperature += dH
                                        c2.temperature += dH
        else:
            # 전문 모드: Arrhenius 식
            for y in range(grid.height):
                for x in range(grid.width):
                    c1 = grid.GetCell(x,y)
                    rxList = self.rxDB.GetReactionsFor(c1.materialID)
                    for dx,dy in directions:
                        nx,ny = x+dx,y+dy
                        if grid.InBounds(nx,ny):
                            c2 = grid.GetCell(nx,ny)
                            for rx in rxList:
                                T = (c1.temperature+c2.temperature)*0.5+273.15
                                rate = rx.A*math.exp(-rx.Ea/(self.R*T))*self.config.reactionPrecision*self.config.simulationSpeed
                                if random.random()<rate*dt:
                                    if rx.products:
                                        c1.materialID = rx.products[0]
                                    dH = rx.deltaH * dt * self.config.reactionPrecision*self.config.simulationSpeed
                                    c1.temperature += dH
                                    c2.temperature += dH

class FluidSolver:
    def __init__(self, matDB, config):
        self.matDB = matDB
        self.config = config
        self.viscosity = 1e-5

    def Solve(self, grid, dt):
        w,h = grid.width, grid.height
        g = 9.81
        for c in grid.cells:
            c.velocityY += g*dt*self.config.simulationSpeed
        # 단순 밀도 기반 정렬
        for y in range(h-1,0,-1):
            for x in range(w):
                c1 = grid.GetCell(x,y)
                c2 = grid.GetCell(x,y-1)
                m1 = self.matDB.GetMaterial(c1.materialID)
                m2 = self.matDB.GetMaterial(c2.materialID)
                if m1.density > m2.density:
                    grid.cells[y*w+x],grid.cells[(y-1)*w+x] = grid.cells[(y-1)*w+x],grid.cells[y*w+x]

class ThermalSolver:
    def __init__(self, matDB, config):
        self.matDB = matDB
        self.config = config

    def Solve(self, grid, dt):
        w,h = grid.width,grid.height
        newTemps = np.zeros((h,w),dtype=float)
        for y in range(h):
            for x in range(w):
                c = grid.GetCell(x,y)
                Tsum=0.0
                weightSum=0.0
                baseMat = self.matDB.GetMaterial(c.materialID)
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        nx, ny = x+dx,y+dy
                        if 0<=nx<w and 0<=ny<h:
                            cc = grid.GetCell(nx,ny)
                            neighMat = self.matDB.GetMaterial(cc.materialID)
                            cond = (baseMat.thermalConductivity+neighMat.thermalConductivity)*0.5
                            Tsum += cc.temperature*cond
                            weightSum += cond
                if weightSum>0:
                    newTemps[y,x] = Tsum/weightSum
                else:
                    newTemps[y,x] = c.temperature

        # 대류 근사
        finalTemps = np.copy(newTemps)
        for y in range(h):
            for x in range(w):
                c = grid.GetCell(x,y)
                vx = c.velocityX*dt*self.config.simulationSpeed
                vy = c.velocityY*dt*self.config.simulationSpeed
                sx = int(round(x - vx))
                sy = int(round(y - vy))
                if 0<=sx<w and 0<=sy<h:
                    finalTemps[y,x] = newTemps[sy,sx]

        for y in range(h):
            for x in range(w):
                grid.GetCell(x,y).temperature = finalTemps[y,x]

#--------------------------------------------
# SimulationManager
#--------------------------------------------
class Config:
    def __init__(self):
        self.expertMode = True
        self.reactionPrecision = 1.0
        self.gridWidth = 100
        self.gridHeight = 100
        self.spawnRate = 1.0
        self.updateIntervalMs = 16
        self.selectedMaterialID = 0
        self.brushSize = 3
        self.screenWidth = 1280
        self.screenHeight = 800
        self.viewZoom = 1.0
        self.viewOffsetX = 0
        self.viewOffsetY = 0
        self.displayMode = "Material"
        self.paused = False
        self.simulationSpeed = 1.0
        self.showTools = True

class SimulationManager:
    def __init__(self, config):
        self.config = config
        self.matDB = MaterialDatabase()
        self.matDB.LoadFromJSON(MATERIALS_JSON)
        self.rxDB = ReactionDatabase()

        self.grid = CellGrid(config.gridWidth,config.gridHeight)
        self.reactionEngine = ReactionEngine(self.matDB, self.rxDB, config)
        self.fluidSolver = FluidSolver(self.matDB, config)
        self.thermalSolver = ThermalSolver(self.matDB, config)
        self.running = True
        self.tools = []
        # 간단한 반응 추가: NaOH + H2SO4 -> Water
        NaOH = self.matDB.nameToID["NaOH"]
        H2SO4 = self.matDB.nameToID["H2SO4"]
        Water = self.matDB.nameToID["Water"]
        rx = Reaction(NaOH,H2SO4,[Water],0.01,50000,-500)
        self.rxDB.AddReaction(rx)

        # Ethanol + O2 -> CO2 (단순)
        Ethanol = self.matDB.nameToID["Ethanol"]
        O2 = self.matDB.nameToID["O2"]
        CO2 = self.matDB.nameToID["CO2"]
        rx2 = Reaction(Ethanol,O2,[CO2],0.05,80000,-800)
        self.rxDB.AddReaction(rx2)

    def Initialize(self):
        # 스포너 설치: 맵 상단에 O2 공급
        O2 = self.matDB.nameToID["O2"]
        self.grid.SetSpawner(self.config.gridWidth//2,0,O2)

        # 툴 설치: Heater 왼쪽 아래
        self.tools.append(Heater("Heater",10,10,5,5))
        self.tools.append(Cooler("Cooler",70,70,5,5))
        self.tools.append(Beaker("Beaker",40,40,10,10))

    def Update(self, dt):
        if self.config.paused:
            return
        dt *= self.config.simulationSpeed
        self.grid.UpdateSpawners(self.config.spawnRate)
        self.reactionEngine.ProcessReactions(self.grid, dt)
        self.fluidSolver.Solve(self.grid, dt)
        self.thermalSolver.Solve(self.grid, dt)
        for tool in self.tools:
            tool.apply(self.grid, self.matDB, dt, self.config.expertMode)

#--------------------------------------------
# UI: SimulationView
#--------------------------------------------
class SimulationView(QOpenGLWidget):
    def __init__(self, simManager):
        super().__init__()
        self.simManager = simManager
        self.setFocusPolicy(Qt.StrongFocus)
    def paintEvent(self, event):
        painter = QPainter(self)
        w,h = self.simManager.grid.width, self.simManager.grid.height
        cw = self.width()/w*self.simManager.config.viewZoom
        ch = self.height()/h*self.simManager.config.viewZoom
        offsetX = self.simManager.config.viewOffsetX
        offsetY = self.simManager.config.viewOffsetY
        displayMode = self.simManager.config.displayMode
        for y in range(h):
            for x in range(w):
                c = self.simManager.grid.GetCell(x,y)
                mat = self.simManager.matDB.GetMaterial(c.materialID)
                if displayMode == "Material":
                    r,g,b = mat.color
                elif displayMode == "Temperature":
                    T = (c.temperature+200)/1200
                    T = max(0,min(1,T))
                    r = T; g=0.0; b=1.0-T
                elif displayMode == "Pressure":
                    p_norm = (c.pressure-100000)/200000
                    p_norm = max(0,min(1,p_norm))
                    r=g=b=p_norm
                else:
                    r,g,b=mat.color
                color = QColor(int(r*255),int(g*255),int(b*255))
                painter.fillRect(int((x+offsetX)*cw),int((y+offsetY)*ch),int(cw+1),int(ch+1), color)

        if self.simManager.config.showTools:
            painter.setPen(QColor(255,255,255))
            for tool in self.simManager.tools:
                painter.drawRect(int((tool.x+offsetX)*cw),int((tool.y+offsetY)*ch),
                                 int(tool.w*cw),int(tool.h*ch))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.paintMaterial(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.paintMaterial(event)

    def paintMaterial(self, event):
        w,h = self.simManager.grid.width, self.simManager.grid.height
        cw = self.width()/w*self.simManager.config.viewZoom
        ch = self.height()/h*self.simManager.config.viewZoom
        offsetX = self.simManager.config.viewOffsetX
        offsetY = self.simManager.config.viewOffsetY
        gridX = int(event.position().x()/cw - offsetX)
        gridY = int(event.position().y()/ch - offsetY)
        if 0<=gridX<w and 0<=gridY<h:
            self.simManager.grid.AddMaterial(gridX,gridY,self.simManager.config.selectedMaterialID,self.simManager.config.brushSize)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()/120
        self.simManager.config.viewZoom *= (1+delta*0.1)
        self.simManager.config.viewZoom = max(0.1,min(10,self.simManager.config.viewZoom))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.simManager.config.viewOffsetY -= 1
        elif event.key() == Qt.Key_S:
            self.simManager.config.viewOffsetY += 1
        elif event.key() == Qt.Key_A:
            self.simManager.config.viewOffsetX -= 1
        elif event.key() == Qt.Key_D:
            self.simManager.config.viewOffsetX += 1

#--------------------------------------------
# MaterialDock: 검색, 트리뷰로 물질 선택
#--------------------------------------------
class MaterialDock(QDockWidget):
    def __init__(self, simManager):
        super().__init__("Materials")
        self.simManager = simManager
        w = QWidget()
        layout = QVBoxLayout(w)

        self.searchEdit = QLineEdit()
        self.searchEdit.setPlaceholderText("Search material...")
        self.searchEdit.textChanged.connect(self.filterMaterials)
        layout.addWidget(self.searchEdit)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        layout.addWidget(self.tree)

        self.populateMaterials()
        self.tree.itemClicked.connect(self.selectMaterial)

        self.setWidget(w)

    def populateMaterials(self):
        # 그룹화 예: Metals, Inorganic, Organic, Gases
        metals = QTreeWidgetItem(["Metals"])
        inorg = QTreeWidgetItem(["Inorganic"])
        org = QTreeWidgetItem(["Organic"])
        gas = QTreeWidgetItem(["Gases"])

        for i,m in enumerate(self.simManager.matDB.materials):
            item = QTreeWidgetItem([m.name])
            # 카테고리 분류 간단히
            if m.boilingPoint<0:
                gas.addChild(item)
            elif m.density>5000:
                metals.addChild(item)
            elif "H2SO4" in m.name or "NaOH" in m.name or "NaCl" in m.name or "SiO2" in m.name:
                inorg.addChild(item)
            elif "Ethanol" in m.name:
                org.addChild(item)
            else:
                inorg.addChild(item)

        self.tree.addTopLevelItem(metals)
        self.tree.addTopLevelItem(inorg)
        self.tree.addTopLevelItem(org)
        self.tree.addTopLevelItem(gas)
        self.tree.expandAll()

    def filterMaterials(self, text):
        # 간단한 필터
        matchText = text.lower()
        def filterItem(item):
            visible = (matchText in item.text(0).lower())
            for i in range(item.childCount()):
                c = item.child(i)
                if filterItem(c):
                    visible = True
            item.setHidden(not visible)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            filterItem(root)

    def selectMaterial(self, item, col):
        name = item.text(0)
        if name in self.simManager.matDB.nameToID:
            self.simManager.config.selectedMaterialID = self.simManager.matDB.nameToID[name]

#--------------------------------------------
# ControlDock: 시뮬레이션 제어
#--------------------------------------------
class ControlDock(QDockWidget):
    def __init__(self, simManager):
        super().__init__("Control")
        self.simManager = simManager
        w = QWidget()
        layout = QVBoxLayout(w)

        self.expertCheck = QCheckBox("Expert Mode")
        self.expertCheck.setChecked(simManager.config.expertMode)
        self.expertCheck.stateChanged.connect(self.toggleExpert)
        layout.addWidget(self.expertCheck)

        self.pauseButton = QPushButton("Pause/Resume")
        self.pauseButton.clicked.connect(self.togglePause)
        layout.addWidget(self.pauseButton)

        layout.addWidget(QLabel("Brush Size"))
        self.brushSlider = QSlider(Qt.Horizontal)
        self.brushSlider.setRange(1,20)
        self.brushSlider.setValue(simManager.config.brushSize)
        self.brushSlider.valueChanged.connect(self.changeBrushSize)
        layout.addWidget(self.brushSlider)

        layout.addWidget(QLabel("Simulation Speed"))
        self.speedSlider = QSlider(Qt.Horizontal)
        self.speedSlider.setRange(1,50)
        self.speedSlider.setValue(int(simManager.config.simulationSpeed*10))
        self.speedSlider.valueChanged.connect(self.changeSimSpeed)
        layout.addWidget(self.speedSlider)

        self.setWidget(w)

    def toggleExpert(self, state):
        self.simManager.config.expertMode = (state == Qt.Checked)

    def togglePause(self):
        self.simManager.config.paused = not self.simManager.config.paused

    def changeBrushSize(self, val):
        self.simManager.config.brushSize = val

    def changeSimSpeed(self, val):
        self.simManager.config.simulationSpeed = val/10.0

#--------------------------------------------
# MainWindow
#--------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, simManager):
        super().__init__()
        self.simManager = simManager
        self.simManager.Initialize()

        self.setWindowTitle("Advanced Laboratory Simulator - Professional Edition")
        self.view = SimulationView(self.simManager)
        self.setCentralWidget(self.view)

        # Dock widgets
        self.materialDock = MaterialDock(self.simManager)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.materialDock)

        self.controlDock = ControlDock(self.simManager)
        self.addDockWidget(Qt.RightDockWidgetArea, self.controlDock)

        # Menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("File")

        loadScriptAction = QAction("Load Script",self)
        loadScriptAction.triggered.connect(self.loadScript)
        fileMenu.addAction(loadScriptAction)

        saveSnapshotAction = QAction("Save Snapshot",self)
        saveSnapshotAction.triggered.connect(self.saveSnapshot)
        fileMenu.addAction(saveSnapshotAction)

        viewMenu = menubar.addMenu("View")
        displayMaterial = QAction("Show Material View",self)
        displayMaterial.triggered.connect(lambda: self.simManager.config.displayMode.__setattr__("material", "Material"))
        viewMenu.addAction(displayMaterial)

        displayTemp = QAction("Show Temperature View",self)
        displayTemp.triggered.connect(lambda: self.setDisplayMode("Temperature"))
        viewMenu.addAction(displayTemp)

        displayPress = QAction("Show Pressure View",self)
        displayPress.triggered.connect(lambda: self.setDisplayMode("Pressure"))
        viewMenu.addAction(displayPress)

        simMenu = menubar.addMenu("Simulation")
        pauseAct = QAction("Pause/Resume",self)
        pauseAct.triggered.connect(self.togglePause)
        simMenu.addAction(pauseAct)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(self.simManager.config.updateIntervalMs)
        self.prevTime = time.time()

    def setDisplayMode(self, mode):
        self.simManager.config.displayMode = mode

    def togglePause(self):
        self.simManager.config.paused = not self.simManager.config.paused

    def tick(self):
        currentTime = time.time()
        dt = currentTime - self.prevTime
        self.prevTime = currentTime
        if self.simManager.running:
            self.simManager.Update(dt)
            self.view.update()

    def loadScript(self):
        path, _ = QFileDialog.getOpenFileName(self,"Load Script","","Python Files (*.py)")
        if path:
            with open(path) as f:
                code = f.read()
            exec(code,{"grid":self.simManager.grid,"matDB":self.simManager.matDB,"rxDB":self.simManager.rxDB})

    def saveSnapshot(self):
        img = self.view.grabFramebuffer()
        path, _ = QFileDialog.getSaveFileName(self,"Save Snapshot","","PNG Files (*.png)")
        if path:
            img.save(path)

#--------------------------------------------
# main
#--------------------------------------------
if __name__=="__main__":
    app = QApplication(sys.argv)
    config = Config()
    simManager = SimulationManager(config)
    window = MainWindow(simManager)
    window.resize(config.screenWidth, config.screenHeight)
    window.show()
    sys.exit(app.exec())
