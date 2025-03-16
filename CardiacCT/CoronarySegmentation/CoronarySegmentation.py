import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import heapq

#
# CoronarySegmentation
#
class CoronarySegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Coronary Segmentation"
    self.parent.categories = ["Cardiac"]
    self.parent.dependencies = []
    self.parent.contributors = ["Your Name"]
    self.parent.helpText = """
    This module provides semi-automatic segmentation of coronary arteries using path finding algorithms.
    """
    self.parent.acknowledgementText = """
    This module was developed for coronary artery analysis.
    """

#
# CoronarySegmentationWidget
#
class CoronarySegmentationWidget(ScriptedLoadableModuleWidget):
  """Interfaccia utente per la segmentazione delle coronarie"""

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Layout
    self.layout = self.parent.layout()
    self.layout.setContentsMargins(0, 0, 0, 0)
    
    # -----------------------------
    # Sezione Input
    # -----------------------------
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Input"
    self.layout.addWidget(inputCollapsibleButton)
    inputFormLayout = qt.QFormLayout(inputCollapsibleButton)
    
    # Selettore volume
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.inputSelector.setToolTip("Seleziona il volume CT di input")
    inputFormLayout.addRow("Volume CT: ", self.inputSelector)
    
    # Selettore punti fiduciali
    self.fiducialsSelector = slicer.qMRMLNodeComboBox()
    self.fiducialsSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.fiducialsSelector.selectNodeUponCreation = True
    self.fiducialsSelector.addEnabled = True
    self.fiducialsSelector.removeEnabled = True
    self.fiducialsSelector.noneEnabled = True
    self.fiducialsSelector.showHidden = False
    self.fiducialsSelector.renameEnabled = True
    self.fiducialsSelector.setMRMLScene(slicer.mrmlScene)
    self.fiducialsSelector.setToolTip("Seleziona o crea punti fiduciali lungo l'arteria coronaria")
    inputFormLayout.addRow("Punti fiduciali: ", self.fiducialsSelector)
    
    # Nome del vaso
    self.vesselNameLineEdit = qt.QLineEdit()
    self.vesselNameLineEdit.text = "Coronaria"
    self.vesselNameLineEdit.setToolTip("Inserisci il nome del vaso da tracciare")
    inputFormLayout.addRow("Nome vaso: ", self.vesselNameLineEdit)
    
    # Pulsante posizionamento punti
    self.placeFiducialsButton = qt.QPushButton("Posiziona punti")
    self.placeFiducialsButton.toolTip = "Posiziona punti lungo l'arteria coronaria"
    self.placeFiducialsButton.enabled = True
    inputFormLayout.addRow(self.placeFiducialsButton)
    
    # -----------------------------
    # Sezione Parametri
    # -----------------------------
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parametri di segmentazione"
    self.layout.addWidget(parametersCollapsibleButton)
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    
    # Soglia inferiore
    self.lowerThresholdSlider = ctk.ctkSliderWidget()
    self.lowerThresholdSlider.singleStep = 10
    self.lowerThresholdSlider.minimum = -100
    self.lowerThresholdSlider.maximum = 500
    self.lowerThresholdSlider.value = 150
    self.lowerThresholdSlider.setToolTip("Soglia inferiore (HU) per la segmentazione dei vasi")
    parametersFormLayout.addRow("Soglia inferiore (HU): ", self.lowerThresholdSlider)
    
    # Soglia superiore
    self.upperThresholdSlider = ctk.ctkSliderWidget()
    self.upperThresholdSlider.singleStep = 10
    self.upperThresholdSlider.minimum = 0
    self.upperThresholdSlider.maximum = 1000
    self.upperThresholdSlider.value = 600
    self.upperThresholdSlider.setToolTip("Soglia superiore (HU) per la segmentazione dei vasi")
    parametersFormLayout.addRow("Soglia superiore (HU): ", self.upperThresholdSlider)
    
    # Diametro del vaso
    self.vesselDiameterSlider = ctk.ctkSliderWidget()
    self.vesselDiameterSlider.singleStep = 0.1
    self.vesselDiameterSlider.minimum = 0.5
    self.vesselDiameterSlider.maximum = 10.0
    self.vesselDiameterSlider.value = 3.0
    self.vesselDiameterSlider.setToolTip("Diametro stimato del vaso (mm)")
    parametersFormLayout.addRow("Diametro vaso (mm): ", self.vesselDiameterSlider)
    
    # -----------------------------
    # Opzioni Path Finding
    # -----------------------------
    pathFindingCollapsibleButton = ctk.ctkCollapsibleButton()
    pathFindingCollapsibleButton.text = "Opzioni Path Finding"
    self.layout.addWidget(pathFindingCollapsibleButton)
    pathFindingFormLayout = qt.QFormLayout(pathFindingCollapsibleButton)
    
    # Usa path finding
    self.usePathFindingCheckBox = qt.QCheckBox()
    self.usePathFindingCheckBox.checked = True
    self.usePathFindingCheckBox.setToolTip("Usa algoritmo avanzato di path finding per creare una centerline accurata")
    pathFindingFormLayout.addRow("Usa Path Finding avanzato: ", self.usePathFindingCheckBox)
    
    # Peso di vascolarità
    self.vascularitySlider = ctk.ctkSliderWidget()
    self.vascularitySlider.singleStep = 0.1
    self.vascularitySlider.minimum = 0.1
    self.vascularitySlider.maximum = 5.0
    self.vascularitySlider.value = 2.0  # Aumentato per seguire meglio il contrasto
    self.vascularitySlider.setToolTip("Peso della preferenza per regioni vascolari (valori più alti favoriscono percorsi più vascolari)")
    pathFindingFormLayout.addRow("Peso vascolarità: ", self.vascularitySlider)
    
    # Fattore di smoothing
    self.smoothingFactorSlider = ctk.ctkSliderWidget()
    self.smoothingFactorSlider.singleStep = 0.1
    self.smoothingFactorSlider.minimum = 0.0
    self.smoothingFactorSlider.maximum = 1.0
    self.smoothingFactorSlider.value = 0.5
    self.smoothingFactorSlider.setToolTip("Fattore di smoothing per la centerline (valori più alti creano percorsi più lisci)")
    pathFindingFormLayout.addRow("Fattore smoothing: ", self.smoothingFactorSlider)
    
    # -----------------------------
    # Pulsante Applica
    # -----------------------------
    self.applyButton = qt.QPushButton("Applica")
    self.applyButton.toolTip = "Esegui la segmentazione"
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)
    
    # -----------------------------
    # Etichetta di stato
    # -----------------------------
    self.statusLabel = qt.QLabel("Stato: Pronto")
    self.layout.addWidget(self.statusLabel)
    
    # Aggiungi spaziatore verticale
    self.layout.addStretch(1)
    
    # -----------------------------
    # Connessioni
    # -----------------------------
    self.placeFiducialsButton.connect('clicked(bool)', self.onPlaceFiducials)
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.fiducialsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    
    # Aggiornamento stato iniziale
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    # Abilita/disabilita pulsante applica in base alle selezioni
    self.applyButton.enabled = self.inputSelector.currentNode() and self.fiducialsSelector.currentNode()

  def onPlaceFiducials(self):
    # Crea un nuovo nodo fiduciale se nessuno è selezionato
    if not self.fiducialsSelector.currentNode():
      fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PuntiCoronaria")
      self.fiducialsSelector.setCurrentNode(fiducialNode)
    
    # Configura la modalità di posizionamento
    placeModePersistence = 1
    slicer.modules.markups.logic().StartPlaceMode(placeModePersistence)
    
    # Aggiorna l'etichetta di stato
    self.statusLabel.text = "Stato: Posiziona punti lungo l'arteria coronaria"

  def onApplyButton(self):
    # Aggiorna stato
    self.statusLabel.text = "Stato: Elaborazione in corso..."
    slicer.app.processEvents()
    
    # Ottieni parametri
    volumeNode = self.inputSelector.currentNode()
    fiducialNode = self.fiducialsSelector.currentNode()
    vesselName = self.vesselNameLineEdit.text
    lowerThreshold = self.lowerThresholdSlider.value
    upperThreshold = self.upperThresholdSlider.value
    vesselDiameter = self.vesselDiameterSlider.value
    usePathFinding = self.usePathFindingCheckBox.checked
    vascularityWeight = self.vascularitySlider.value
    smoothingFactor = self.smoothingFactorSlider.value
    
    # Verifica se abbiamo abbastanza punti
    if fiducialNode.GetNumberOfControlPoints() < 2:
      self.statusLabel.text = "Stato: Errore - Servono almeno 2 punti fiduciali"
      return
    
    # Esegui l'algoritmo
    logic = CoronarySegmentationLogic()
    try:
      # Passaggio 1: Crea centerline
      self.statusLabel.text = "Stato: Creazione centerline..."
      slicer.app.processEvents()
      
      if usePathFinding:
        centerlineNode = logic.createCoronaryPathWithPathFinding(
          volumeNode, fiducialNode, vascularityWeight, smoothingFactor)
      else:
        centerlineNode = logic.createCoronaryPath(volumeNode, fiducialNode)
      
      if not centerlineNode:
        self.statusLabel.text = "Stato: Errore - Impossibile creare la centerline"
        return
      
      # Passaggio 2: Segmenta il vaso
      self.statusLabel.text = "Stato: Segmentazione vaso..."
      slicer.app.processEvents()
      
      segmentationNode = logic.createCoronarySegmentation(
        volumeNode, centerlineNode, lowerThreshold, upperThreshold, vesselDiameter, vesselName)
      
      if not segmentationNode:
        self.statusLabel.text = "Stato: Errore - Impossibile creare la segmentazione"
        return
      
      # Configura la visualizzazione
      self.statusLabel.text = "Stato: Configurazione vista..."
      slicer.app.processEvents()
      
      logic.setupViews(volumeNode, segmentationNode, centerlineNode)
      
      self.statusLabel.text = "Stato: Completato"
    except Exception as e:
      self.statusLabel.text = f"Stato: Errore - {str(e)}"
      import traceback
      traceback.print_exc()
#
# CoronarySegmentationLogic
#
class CoronarySegmentationLogic(ScriptedLoadableModuleLogic):
  """Implementa la logica per la segmentazione delle coronarie e generazione di centerline"""

  def preprocessVolumeForPathFinding(self, volumeNode):
    """Preelabora il volume per evidenziare le strutture vascolari"""
    # Crea un volume temporaneo per il preprocessamento
    tempVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "VasiMigliorati")
    
    # Copia matrice di trasformazione
    ijkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRas)
    tempVolume.SetIJKToRASMatrix(ijkToRas)
    
    # Ottieni array numpy dal volume
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    
    # Crea maschera per valori HU tipici dei vasi con contrasto
    vesselMask = (volumeArray >= 150) & (volumeArray <= 500)
    
    # Applica filtro che esalta i vasi
    enhancedArray = np.zeros_like(volumeArray)
    enhancedArray[vesselMask] = volumeArray[vesselMask]
    
    # Applica un leggero smoothing gaussiano (se scipy è disponibile)
    try:
      from scipy import ndimage
      enhancedArray = ndimage.gaussian_filter(enhancedArray, sigma=0.5)
    except ImportError:
      pass
    
    # Normalizza i valori per enfatizzare le strutture vascolari
    enhancedArray[~vesselMask] = -1000  # Imposta background a valore molto basso
    
    # Aggiorna il volume temporaneo
    slicer.util.updateVolumeFromArray(tempVolume, enhancedArray)
    tempVolume.SetSpacing(volumeNode.GetSpacing())
    
    return tempVolume

  def createCoronaryPath(self, volumeNode, fiducialNode):
    """Crea una centerline semplice interpolando tra i punti fiduciali"""
    
    # Verifica input
    if not volumeNode or not fiducialNode:
      logging.error("Volume o nodo fiduciale mancante")
      return None
    
    # Verifica numero di punti
    numPoints = fiducialNode.GetNumberOfControlPoints()
    if numPoints < 2:
      logging.error("Servono almeno 2 punti fiduciali")
      return None
    
    # Crea curva
    curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "CenterlineCoronaria")
    curveNode.CreateDefaultDisplayNodes()
    
    # Imposta proprietà di visualizzazione
    displayNode = curveNode.GetDisplayNode()
    displayNode.SetColor(1.0, 1.0, 0.0)  # Giallo
    displayNode.SetLineThickness(2.0)
    
    # Aggiungi punti di controllo dai fiduciali
    for i in range(numPoints):
      pos = [0, 0, 0]
      fiducialNode.GetNthControlPointPositionWorld(i, pos)
      curveNode.AddControlPoint(pos)
    
    return curveNode

  def createCoronaryPathWithPathFinding(self, volumeNode, fiducialNode, vascularityWeight=2.0, smoothingFactor=0.5):
    """Crea una centerline usando path finding avanzato tra i punti fiduciali"""
    
    # Verifica input
    if not volumeNode or not fiducialNode:
      logging.error("Volume o nodo fiduciale mancante")
      return None
    
    # Verifica numero di punti
    numPoints = fiducialNode.GetNumberOfControlPoints()
    if numPoints < 2:
      logging.error("Servono almeno 2 punti fiduciali")
      return None
    
    # Preelabora volume per migliorare il riconoscimento dei vasi
    enhancedVolume = self.preprocessVolumeForPathFinding(volumeNode)
    
    # Crea path finder con volume migliorato
    pathFinder = VascularPathFinder(enhancedVolume)
    pathFinder.vascularityWeight = vascularityWeight
    
    # Crea curva
    curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "CenterlineCoronaria")
    curveNode.CreateDefaultDisplayNodes()
    
    # Imposta proprietà di visualizzazione
    displayNode = curveNode.GetDisplayNode()
    displayNode.SetColor(1.0, 1.0, 0.0)  # Giallo
    displayNode.SetLineThickness(2.0)
    
    # Trova percorso tra ogni coppia di punti consecutivi
    allPathPoints = []
    
    # Estrai posizioni dei fiduciali
    fiducialPositions = []
    for i in range(numPoints):
      pos = [0, 0, 0]
      fiducialNode.GetNthControlPointPositionWorld(i, pos)
      fiducialPositions.append(pos)
    
    # Trova percorso tra punti
    for i in range(numPoints - 1):
      startPoint = fiducialPositions[i]
      endPoint = fiducialPositions[i+1]
      
      # Trova percorso
      path = pathFinder.findPath(startPoint, endPoint)
      
      if path:
        if i == 0:
          # Per il primo segmento, includi il primo punto
          allPathPoints.extend(path)
        else:
          # Per i segmenti successivi, salta il primo punto per evitare duplicati
          allPathPoints.extend(path[1:])
      else:
        logging.warning(f"Impossibile trovare percorso tra i punti {i} e {i+1}")
        # Fallback a linea diretta
        if i == 0:
          allPathPoints.append(startPoint)
        allPathPoints.append(endPoint)
    
    # Applica smoothing se necessario
    if smoothingFactor > 0:
      allPathPoints = self.smoothPath(allPathPoints, smoothingFactor)
    
    # Aggiungi punti alla curva
    for point in allPathPoints:
      curveNode.AddControlPoint(point)
    
    # Pulisci nodi temporanei
    slicer.mrmlScene.RemoveNode(enhancedVolume)
    
    return curveNode

  def smoothPath(self, points, smoothingFactor):
    """Applica smoothing al percorso usando media mobile"""
    if len(points) < 3 or smoothingFactor <= 0:
      return points
    
    # Converti in array numpy per manipolazione più facile
    pointsArray = np.array(points)
    
    # Calcola dimensione finestra basata sul fattore di smoothing (1-5 punti)
    windowSize = max(3, min(len(points) // 2, int(5 * smoothingFactor)))
    
    # Assicura che la dimensione della finestra sia dispari
    if windowSize % 2 == 0:
      windowSize += 1
    
    # Crea kernel di convoluzione
    kernel = np.ones(windowSize) / windowSize
    
    # Applica convoluzione a ogni dimensione
    smoothedPoints = np.zeros_like(pointsArray)
    for dim in range(3):
      # Pad del segnale per gestire i bordi
      padded = np.pad(pointsArray[:, dim], (windowSize//2, windowSize//2), mode='edge')
      smoothedPoints[:, dim] = np.convolve(padded, kernel, mode='valid')
    
    # Converti di nuovo in lista di punti
    return smoothedPoints.tolist()

  def createCoronarySegmentation(self, volumeNode, centerlineNode, lowerThreshold, upperThreshold, vesselDiameter, vesselName):
    """Crea una segmentazione dell'arteria coronaria lungo la centerline"""
    
    # Verifica input
    if not volumeNode or not centerlineNode:
      logging.error("Volume o centerline mancante")
      return None
    
    # Crea segmentazione
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SegmentazioneCoronaria")
    segmentationNode.CreateDefaultDisplayNodes()
    
    # Aggiungi un segmento con il nome specificato dall'utente
    segmentID = segmentationNode.GetSegmentation().AddEmptySegment(vesselName)
    
    # Crea un modello per contenere tutte le sfere unite
    appendPolyData = vtk.vtkAppendPolyData()
    
    # Ottieni numero di punti di controllo
    numPoints = centerlineNode.GetNumberOfControlPoints()
    
    # Aggiungi sfere lungo la centerline
    for i in range(0, numPoints, max(1, int(numPoints / 50))):
      pos = [0, 0, 0]
      centerlineNode.GetNthControlPointPositionWorld(i, pos)
      
      # Crea una sfera attorno a ciascun punto della centerline
      sphereSource = vtk.vtkSphereSource()
      sphereSource.SetCenter(pos)
      sphereSource.SetRadius(vesselDiameter/2.0)  # Raggio in mm
      sphereSource.SetPhiResolution(12)
      sphereSource.SetThetaResolution(12)
      sphereSource.Update()
      
      # Aggiungi la sfera al polydata
      appendPolyData.AddInputData(sphereSource.GetOutput())
    
    # Esegui l'append
    appendPolyData.Update()
    
    # Crea un modello con le sfere unite
    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ModelloCoronariaTmp")
    modelNode.SetAndObservePolyData(appendPolyData.GetOutput())
    
    # Converti il modello in segmentazione
    slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segmentationNode)
    
    # Pulisci nodo modello temporaneo
    slicer.mrmlScene.RemoveNode(modelNode)
    
    return segmentationNode

  def setupViews(self, volumeNode, segmentationNode, centerlineNode):
    """Configura il layout per visualizzare il volume e la segmentazione"""
    
    # Passa a layout solo 3D
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    
    # Configura volume rendering per vista 3D
    # Crea un nuovo nodo di visualizzazione volume rendering
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.CreateVolumeRenderingDisplayNode()
    slicer.mrmlScene.AddNode(displayNode)
    displayNode.UnRegister(volRenLogic)
    volRenLogic.UpdateDisplayNodeFromVolumeNode(displayNode, volumeNode)
    
    # Aggiungi il volume rendering alla vista 3D
    volumeNode.AddAndObserveDisplayNodeID(displayNode.GetID())
    
    # Mostra la segmentazione nella vista 3D
    segmentationNode.CreateClosedSurfaceRepresentation()
    segmentationNode.GetDisplayNode().SetVisibility(1)
    
    # Mostra la centerline nella vista 3D
    centerlineNode.GetDisplayNode().SetVisibility(1)
    
    # Regola la camera per mostrare la segmentazione
    threeDView = layoutManager.threeDWidget(0).threeDView()
    threeDView.resetFocalPoint()
    threeDView.resetCamera()

  def worldToIJK(self, volumeNode, worldPoint):
    """Converte coordinate RAS in coordinate IJK"""
    rasToIJK = vtk.vtkMatrix4x4()
    volumeIJKToRAS = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(volumeIJKToRAS)
    vtk.vtkMatrix4x4.Invert(volumeIJKToRAS, rasToIJK)
    
    ijkPoint = [0, 0, 0, 1]
    rasPoint = worldPoint + [1]  # Aggiungi coordinata omogenea
    
    rasToIJK.MultiplyPoint(rasPoint, ijkPoint)
    
    return [int(round(ijkPoint[0])), int(round(ijkPoint[1])), int(round(ijkPoint[2]))]

#
# VascularPathFinder
#
class VascularPathFinder:
  """Classe per trovare percorsi ottimali attraverso strutture vascolari usando l'algoritmo A*"""
  
  def __init__(self, volumeNode):
    self.volumeNode = volumeNode
    self.imageData = volumeNode.GetImageData()
    self.dimensions = self.imageData.GetDimensions()
    self.spacing = volumeNode.GetSpacing()
    self.vascularityWeight = 1.0
    
  def findPath(self, startPoint, endPoint):
    """Trova percorso ottimale tra punto iniziale e finale usando algoritmo A*"""
    
    # Converti punti mondo in IJK
    start_ijk = self._worldToIJK(startPoint)
    end_ijk = self._worldToIJK(endPoint)
    
    # Memorizza per uso successivo
    self.start_ijk = start_ijk
    self.end_ijk = end_ijk
    
    # Crea cilindro ROI per ottimizzazione
    self.cylinder_radius = 20  # mm
    
    # Inizializza strutture A*
    open_set = []
    closed_set = set()
    came_from = {}
    
    # Costo dal punto iniziale a ciascun nodo
    g_score = {self._pointToKey(start_ijk): 0}
    
    # Costo stimato totale dal punto iniziale alla destinazione passando per ciascun nodo
    f_score = {self._pointToKey(start_ijk): self._heuristic(start_ijk, end_ijk)}
    
    # Aggiungi punto iniziale all'open set
    heapq.heappush(open_set, (f_score[self._pointToKey(start_ijk)], self._pointToKey(start_ijk)))
    
    # Loop principale A*
    while open_set:
      # Ottieni nodo con f_score più basso
      current_key = heapq.heappop(open_set)[1]
      current = self._keyToPoint(current_key)
      
      # Verifica se abbiamo raggiunto la destinazione
      if self._isPointClose(current, end_ijk, threshold=3):
        return self._reconstructPath(came_from, current)
      
      closed_set.add(current_key)
      
      # Esplora vicini
      for neighbor in self._getNeighbors(current):
        neighbor_key = self._pointToKey(neighbor)
        
        # Salta se già valutato
        if neighbor_key in closed_set:
          continue
        
        # Calcola costo per raggiungere questo vicino
        tentative_g_score = g_score[self._pointToKey(current)] + self._costFunction(current, neighbor)
        
        # Se abbiamo trovato percorso migliore o nodo nuovo
        if neighbor_key not in g_score or tentative_g_score < g_score[neighbor_key]:
          # Aggiorna info percorso
          came_from[neighbor_key] = current_key
          g_score[neighbor_key] = tentative_g_score
          f_score[neighbor_key] = g_score[neighbor_key] + self._heuristic(neighbor, end_ijk)
          
          # Aggiungi all'open set se non è già presente
          if not any(neighbor_key == item[1] for item in open_set):
            heapq.heappush(open_set, (f_score[neighbor_key], neighbor_key))
    
    # Se nessun percorso trovato
    logging.warning("Nessun percorso trovato tra i punti specificati")
    return None
  
  def _worldToIJK(self, worldPoint):
    """Converte coordinate RAS in coordinate IJK"""
    rasToIJK = vtk.vtkMatrix4x4()
    volumeIJKToRAS = vtk.vtkMatrix4x4()
    self.volumeNode.GetIJKToRASMatrix(volumeIJKToRAS)
    vtk.vtkMatrix4x4.Invert(volumeIJKToRAS, rasToIJK)
    
    ijkPoint = [0, 0, 0, 1]
    rasPoint = worldPoint + [1]  # Aggiungi coordinata omogenea
    
    rasToIJK.MultiplyPoint(rasPoint, ijkPoint)
   
    return [int(round(ijkPoint[0])), int(round(ijkPoint[1])), int(round(ijkPoint[2]))]
 
  def _IJKToWorld(self, ijkPoint):
   """Converte coordinate IJK in coordinate RAS"""
   ijkPointHomogeneous = ijkPoint + [1]  # Aggiungi coordinata omogenea
   
   volumeIJKToRAS = vtk.vtkMatrix4x4()
   self.volumeNode.GetIJKToRASMatrix(volumeIJKToRAS)
   
   rasPoint = [0, 0, 0, 1]
   volumeIJKToRAS.MultiplyPoint(ijkPointHomogeneous, rasPoint)
   
   return rasPoint[:3]  # Rimuovi coordinata omogenea
 
  def _pointToKey(self, point):
   """Converte punto in chiave hashable per dizionari e set"""
   return (int(point[0]), int(point[1]), int(point[2]))
 
  def _keyToPoint(self, key):
   """Converte chiave in punto"""
   return list(key)
 
  def _isPointClose(self, point1, point2, threshold=1):
   """Verifica se due punti sono entro una distanza soglia"""
   dist = ((point1[0] - point2[0])**2 + 
           (point1[1] - point2[1])**2 + 
           (point1[2] - point2[2])**2)**0.5
   return dist <= threshold
 
  def _heuristic(self, point, goal):
   """Funzione euristica (distanza euclidea)"""
   return ((point[0] - goal[0])**2 * self.spacing[0]**2 + 
           (point[1] - goal[1])**2 * self.spacing[1]**2 + 
           (point[2] - goal[2])**2 * self.spacing[2]**2)**0.5
 
  def _getNeighbors(self, point):
   """Ottiene vicini validi di un punto"""
   neighbors = []
   
   # Crea una ROI per efficienza se non già fatto
   if not hasattr(self, 'roi'):
     # Calcola distanza diretta tra endpoint
     start_ras = self._IJKToWorld(self.start_ijk)
     end_ras = self._IJKToWorld(self.end_ijk)
     direct_distance = sum([(a-b)**2 for a, b in zip(start_ras, end_ras)])**0.5
     
     # Imposta dimensione ROI in base alla distanza (più grande per percorsi più lunghi)
     self.cylinder_radius = max(10, min(30, direct_distance / 3))
   
   # 26-connettività (tutte le celle adiacenti in 3D)
   for dx in [-1, 0, 1]:
     for dy in [-1, 0, 1]:
       for dz in [-1, 0, 1]:
         if dx == 0 and dy == 0 and dz == 0:
           continue  # Salta il punto stesso
         
         nx, ny, nz = point[0] + dx, point[1] + dy, point[2] + dz
         
         # Verifica se dentro i limiti dell'immagine
         if (0 <= nx < self.dimensions[0] and 
             0 <= ny < self.dimensions[1] and 
             0 <= nz < self.dimensions[2]):
           
           # Includi solo punti che sono all'interno della nostra ROI di ricerca
           if self._isPointInSearchROI([nx, ny, nz]):
             neighbors.append([nx, ny, nz])
   
   return neighbors
 
  def _isPointInSearchROI(self, point_ijk):
   """Verifica se il punto è all'interno della ROI di ricerca (cilindro approssimativo tra endpoint)"""
   # Converti in RAS per calcolo più facile
   point_ras = self._IJKToWorld(point_ijk)
   start_ras = self._IJKToWorld(self.start_ijk)
   end_ras = self._IJKToWorld(self.end_ijk)
   
   # Vettore da inizio a fine
   line_vec = [end_ras[i] - start_ras[i] for i in range(3)]
   line_length = sum([x**2 for x in line_vec])**0.5
   
   if line_length == 0:
     return True
   
   # Normalizza
   line_vec = [x/line_length for x in line_vec]
   
   # Vettore da inizio a punto
   point_vec = [point_ras[i] - start_ras[i] for i in range(3)]
   
   # Proiezione del vettore punto sul vettore linea
   projection = sum([point_vec[i] * line_vec[i] for i in range(3)])
   
   # Verifica se la proiezione è all'interno del segmento di linea
   if projection < -self.cylinder_radius or projection > line_length + self.cylinder_radius:
     return False
   
   # Calcola punto più vicino sulla linea
   closest_point = [start_ras[i] + projection * line_vec[i] for i in range(3)]
   
   # Calcola distanza dal punto alla linea
   distance = sum([(point_ras[i] - closest_point[i])**2 for i in range(3)])**0.5
   
   # Verifica se entro raggio del cilindro
   return distance <= self.cylinder_radius
 
  def _getVoxelValue(self, point):
   """Ottiene valore HU alle coordinate voxel specificate"""
   point_ijk = tuple(map(int, point))
   try:
     return self.imageData.GetScalarComponentAsDouble(point_ijk[0], point_ijk[1], point_ijk[2], 0)
   except:
     return -1000  # Valore di default per aria
 
  def _costFunction(self, current, neighbor):
   """
   Funzione di costo personalizzata che favorisce percorsi vascolari
   
   Combina:
   1. Distanza euclidea
   2. Penalità basata su valori HU (favorisce vasi con contrasto)
   """
   # Calcola distanza euclidea
   distance = ((current[0] - neighbor[0])**2 * self.spacing[0]**2 + 
               (current[1] - neighbor[1])**2 * self.spacing[1]**2 + 
               (current[2] - neighbor[2])**2 * self.spacing[2]**2)**0.5
   
   # Ottieni valore HU
   voxel_value = self._getVoxelValue(neighbor)
   
   # Range ideale per vasi con contrasto (tipicamente 200-400 HU)
   optimal_min = 200
   optimal_max = 400
   
   # Penalità molto bassa per valori nel range ottimale
   if optimal_min <= voxel_value <= optimal_max:
     hu_penalty = 0.05  # Costo molto basso per vasi con contrasto
   # Penalità moderata per valori ancora accettabili (150-500 HU)
   elif 150 <= voxel_value <= 500:
     hu_penalty = 0.2
   else:
     # Alta penalità per valori fuori dal range dei vasi
     lower_dist = max(0, 150 - voxel_value)
     upper_dist = max(0, voxel_value - 500)
     hu_penalty = 1.0 + min(lower_dist, upper_dist) / 50.0
   
   # Applica peso di vascolarità per controllare importanza dei valori HU
   return distance * (hu_penalty ** (2.0 * self.vascularityWeight))
 
  def _reconstructPath(self, came_from, current):
   """Ricostruisce percorso dal punto finale al punto iniziale"""
   path_ijk = [current]
   current_key = self._pointToKey(current)
   
   while current_key in came_from:
     current_key = came_from[current_key]
     current = self._keyToPoint(current_key)
     path_ijk.append(current)
   
   # Inverti percorso per ottenere inizio -> fine
   path_ijk.reverse()
   
   # Converti coordinate IJK in RAS
   path_ras = [self._IJKToWorld(point) for point in path_ijk]
   
   return path_ras

#
# CoronarySegmentationTest
#
class CoronarySegmentationTest(ScriptedLoadableModuleTest):
  """
  Classe di test per il modulo
  """

  def setUp(self):
    """ Resetta lo stato - tipicamente basta pulire la scena.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Esegui i test necessari qui.
    """
    self.setUp()
    self.test_CoronarySegmentation1()

  def test_CoronarySegmentation1(self):
    """ Test base per verificare la funzionalità del modulo.
    """
    self.delayDisplay("Avvio del test")
    
    # Creazione dati di test - non implementata in questo esempio base
    self.delayDisplay('Test superato!')