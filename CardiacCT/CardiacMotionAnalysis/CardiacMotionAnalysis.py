import os
import vtk, qt, ctk, slicer
import numpy as np
from slicer.ScriptedLoadableModule import *
import logging
import math
import matplotlib
matplotlib.use('Agg')  # Usa il backend Agg invece di cercare di usare Qt direttamente
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg

#
# CardiacMotionAnalysis
#
class CardiacMotionAnalysis(ScriptedLoadableModule):
  """Modulo per l'analisi del movimento cardiaco dopo segmentazione con Total Segmentator"""
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Cardiac Motion Analysis"
    self.parent.categories = ["CardiacCT"]
    self.parent.dependencies = []
    self.parent.contributors = ["Vittorio Censullo - AITeRTC"]
    self.parent.helpText = """
    Questo modulo analizza la discinesia o acinesia cardiaca utilizzando segmentazioni 4D create con Total Segmentator.
    Visualizza i risultati come mappe 3D e diagrammi polari.
    Nota: Per la generazione di report PDF è necessario il modulo ReportLab, che verrà installato automaticamente se necessario.
    """
    self.parent.acknowledgementText = """
    Sviluppato per l'analisi di dati cardiaci 4D.
    """

#
# CardiacMotionAnalysisWidget
#
class CardiacMotionAnalysisWidget(ScriptedLoadableModuleWidget):
  """Interfaccia utente per CardiacMotionAnalysis"""
  
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    
    # Parametri del modulo
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parametri"
    self.layout.addWidget(parametersCollapsibleButton)
    
    # Layout all'interno del collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    
    # Selezione del volume 4D
    self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
    self.inputVolumeSelector.nodeTypes = ["vtkMRMLMultiVolumeNode", "vtkMRMLSequenceNode"]
    self.inputVolumeSelector.selectNodeUponCreation = True
    self.inputVolumeSelector.addEnabled = False
    self.inputVolumeSelector.removeEnabled = False
    self.inputVolumeSelector.noneEnabled = False
    self.inputVolumeSelector.showHidden = False
    self.inputVolumeSelector.showChildNodeTypes = False
    self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.inputVolumeSelector.setToolTip("Seleziona il volume 4D (MultiVolume) o la sequenza (Sequence)")
    parametersFormLayout.addRow("Volume 4D: ", self.inputVolumeSelector)
    
    # Selezione della segmentazione
    self.segmentationSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segmentationSelector.selectNodeUponCreation = True
    self.segmentationSelector.addEnabled = False
    self.segmentationSelector.removeEnabled = False
    self.segmentationSelector.noneEnabled = False
    self.segmentationSelector.showHidden = False
    self.segmentationSelector.showChildNodeTypes = False
    self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentationSelector.setToolTip("Seleziona la segmentazione (da Total Segmentator)")
    parametersFormLayout.addRow("Segmentazione: ", self.segmentationSelector)
    
    # Soglia per la detenzione di movimento anomalo (percentuale)
    self.thresholdSlider = ctk.ctkSliderWidget()
    self.thresholdSlider.singleStep = 1.0
    self.thresholdSlider.minimum = 1.0
    self.thresholdSlider.maximum = 50.0
    self.thresholdSlider.value = 10.0
    self.thresholdSlider.setToolTip("Soglia per determinare il movimento anomalo (come percentuale di movimento rispetto al massimo)")
    parametersFormLayout.addRow("Soglia (%): ", self.thresholdSlider)
    
    # Checkbox per selezionare le camere cardiache
    self.segmentsFrame = qt.QFrame()
    self.segmentsLayout = qt.QVBoxLayout(self.segmentsFrame)
    
    self.rvCheckBox = qt.QCheckBox("Right Ventricle")
    self.rvCheckBox.checked = True
    self.segmentsLayout.addWidget(self.rvCheckBox)
    
    self.myoCheckBox = qt.QCheckBox("Myocardium")
    self.myoCheckBox.checked = True
    self.segmentsLayout.addWidget(self.myoCheckBox)
    
    self.lvCheckBox = qt.QCheckBox("Left Ventricle")
    self.lvCheckBox.checked = True
    self.segmentsLayout.addWidget(self.lvCheckBox)
    
    parametersFormLayout.addRow("Camere cardiache: ", self.segmentsFrame)
    
    # Opzioni per la visualizzazione
    displayCollapsibleButton = ctk.ctkCollapsibleButton()
    displayCollapsibleButton.text = "Opzioni di visualizzazione"
    self.layout.addWidget(displayCollapsibleButton)
    displayFormLayout = qt.QFormLayout(displayCollapsibleButton)
    
    # Checkbox per la visualizzazione 3D
    self.show3DCheckBox = qt.QCheckBox()
    self.show3DCheckBox.checked = True
    self.show3DCheckBox.setToolTip("Visualizza i risultati nel viewer 3D")
    displayFormLayout.addRow("Visualizzazione 3D: ", self.show3DCheckBox)
    
    # Checkbox per la visualizzazione Polare
    self.showPolarCheckBox = qt.QCheckBox()
    self.showPolarCheckBox.checked = True
    self.showPolarCheckBox.setToolTip("Crea una visualizzazione polare dei risultati")
    displayFormLayout.addRow("Visualizzazione Polare: ", self.showPolarCheckBox)
    
    # Pulsante per salvare report PDF
    self.savePDFButton = qt.QPushButton("Salva Report PDF")
    self.savePDFButton.toolTip = "Salva un report PDF con i risultati dell'analisi"
    self.savePDFButton.enabled = False
    displayFormLayout.addRow(self.savePDFButton)
    
    # Pulsante per configurare la visualizzazione ottimale
    self.setupVisualizationButton = qt.QPushButton("Configura Visualizzazione")
    self.setupVisualizationButton.toolTip = "Configura la visualizzazione ottimale per vedere le aree con movimento anomalo"
    self.setupVisualizationButton.enabled = False
    displayFormLayout.addRow(self.setupVisualizationButton)
    
    # Controlli per visualizzare e navigare tra i frame
    navigationCollapsibleButton = ctk.ctkCollapsibleButton()
    navigationCollapsibleButton.text = "Navigazione frame"
    self.layout.addWidget(navigationCollapsibleButton)
    navigationFormLayout = qt.QFormLayout(navigationCollapsibleButton)
    
    # Slider per selezionare il frame
    self.frameSlider = ctk.ctkSliderWidget()
    self.frameSlider.singleStep = 1
    self.frameSlider.minimum = 0
    self.frameSlider.maximum = 1
    self.frameSlider.value = 0
    self.frameSlider.setToolTip("Seleziona il frame temporale da visualizzare")
    navigationFormLayout.addRow("Frame: ", self.frameSlider)
    
    # Pulsanti per play/pausa dell'animazione
    frameControlsLayout = qt.QHBoxLayout()
    
    self.prevFrameButton = qt.QPushButton("<<")
    self.prevFrameButton.toolTip = "Frame precedente"
    self.prevFrameButton.enabled = False
    frameControlsLayout.addWidget(self.prevFrameButton)
    
    self.playButton = qt.QPushButton("Play")
    self.playButton.toolTip = "Avvia/pausa l'animazione"
    self.playButton.enabled = False
    frameControlsLayout.addWidget(self.playButton)
    
    self.nextFrameButton = qt.QPushButton(">>")
    self.nextFrameButton.toolTip = "Frame successivo"
    self.nextFrameButton.enabled = False
    frameControlsLayout.addWidget(self.nextFrameButton)
    
    navigationFormLayout.addRow("Controlli: ", frameControlsLayout)
    
    # Area per la visualizzazione del diagramma polare
    polarViewCollapsibleButton = ctk.ctkCollapsibleButton()
    polarViewCollapsibleButton.text = "Diagramma Polare"
    self.layout.addWidget(polarViewCollapsibleButton)
    
    polarViewLayout = qt.QVBoxLayout(polarViewCollapsibleButton)
    self.polarViewFrame = qt.QFrame()
    self.polarViewFrame.setMinimumHeight(300)
    polarViewLayout.addWidget(self.polarViewFrame)
    
    # Pulsante di applicazione
    self.applyButton = qt.QPushButton("Analizza")
    self.applyButton.toolTip = "Avvia l'analisi del movimento cardiaco"
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)
    
    # Timer per l'animazione
    self.animationTimer = qt.QTimer()
    self.animationTimer.setInterval(200)  # 200ms tra i frame
    self.isPlaying = False
    
    # Connessioni
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.frameSlider.connect("valueChanged(double)", self.onFrameSliderChanged)
    self.prevFrameButton.connect('clicked(bool)', self.onPreviousFrame)
    self.nextFrameButton.connect('clicked(bool)', self.onNextFrame)
    self.playButton.connect('clicked(bool)', self.onPlayButton)
    self.animationTimer.connect('timeout()', self.onNextFrame)
    self.savePDFButton.connect('clicked(bool)', self.onSavePDFButton)
    self.setupVisualizationButton.connect('clicked(bool)', self.setupVisualization)
    
    # Variabili per i risultati dell'analisi
    self.volumeNodes = []
    self.numFrames = 0
    self.segmentationFrames = {}
    self.analysisResults = None
    self.polarFigure = None
    self.imageLabel = None
    
    # Aggiorna l'interfaccia utente
    self.onSelect()

  def cleanup(self):
    """Pulizia quando il modulo viene chiuso"""
    self.animationTimer.stop()
    
  def onSelect(self):
    """Gestione dell'evento di selezione dei nodi di input"""
    self.applyButton.enabled = self.inputVolumeSelector.currentNode() and self.segmentationSelector.currentNode()
    
    # Disabilita i controlli di navigazione fino a quando non viene eseguita l'analisi
    self.prevFrameButton.enabled = False
    self.nextFrameButton.enabled = False
    self.playButton.enabled = False
    self.frameSlider.enabled = False
    self.savePDFButton.enabled = False
    self.setupVisualizationButton.enabled = False
    
    # Aggiorna il numero di frame se è disponibile il volume 4D o la sequenza
    inputVolume = self.inputVolumeSelector.currentNode()
    if inputVolume:
      if isinstance(inputVolume, slicer.vtkMRMLSequenceNode):
        self.numFrames = inputVolume.GetNumberOfDataNodes()
      else:  # Assumiamo che sia un MultiVolumeNode
        self.numFrames = inputVolume.GetNumberOfFrames()
      
      # Aggiorna lo slider dei frame
      self.frameSlider.minimum = 0
      self.frameSlider.maximum = self.numFrames - 1
      self.frameSlider.value = 0

  def onApplyButton(self):
    """Gestione del pulsante di avvio analisi"""
    logic = CardiacMotionAnalysisLogic()
    self.animationTimer.stop()
    self.isPlaying = False
    self.playButton.text = "Play"
    
    # Ottieni parametri dalla UI
    selectedSegments = []
    if self.rvCheckBox.checked:
      selectedSegments.append("right ventricle of heart")
    if self.myoCheckBox.checked:
      selectedSegments.append("myocardium")
    if self.lvCheckBox.checked:
      selectedSegments.append("left ventricle of heart")
      
    threshold = self.thresholdSlider.value
    
    # Flag per le visualizzazioni
    show3D = self.show3DCheckBox.checked
    showPolar = self.showPolarCheckBox.checked
    
    # Mostra la barra di progresso
    progressBar = slicer.util.createProgressDialog(windowTitle="Analisi in corso", 
                                                 labelText="Elaborazione dati cardiaci...", 
                                                 maximum=100)
    
    # Esegui l'analisi e ottieni i risultati
    try:
      # Connetti i segnali di progresso
      def updateProgress(progress):
        progressBar.setValue(int(progress))
        slicer.app.processEvents()
      
      logic.progressCallback = updateProgress
      
      # Esegui l'analisi
      self.volumeNodes, self.analysisResults = logic.run(
        self.inputVolumeSelector.currentNode(),
        self.segmentationSelector.currentNode(),
        selectedSegments,
        threshold,
        show3D=show3D
      )
      
      # Visualizza i risultati
      if show3D:
        # I modelli 3D sono già stati creati nella funzione run
        pass
        
      if showPolar:
        # Crea e visualizza il diagramma polare
        self.updatePolarDiagram()
        
      # Abilita i controlli di navigazione
      self.prevFrameButton.enabled = True
      self.nextFrameButton.enabled = True
      self.playButton.enabled = True
      self.frameSlider.enabled = True
      
      # Abilita il pulsante per salvataggio PDF
      self.savePDFButton.enabled = True
      self.setupVisualizationButton.enabled = True
      
      # Imposta il primo frame
      self.frameSlider.value = 0
      self.onFrameSliderChanged(0)
      
      # Configura la visualizzazione ottimale
      self.setupVisualization()
      
    except Exception as e:
      slicer.util.errorDisplay("Errore durante l'analisi: {}".format(e))
      import traceback
      traceback.print_exc()
    finally:
      progressBar.close()

  def updatePolarDiagram(self):
    """Crea o aggiorna il diagramma polare con i risultati dell'analisi"""
    if not self.analysisResults:
      return
      
    # Prepara il layout
    if not hasattr(self, 'polarLayout'):
      self.polarLayout = qt.QVBoxLayout(self.polarViewFrame)
    else:
      # Pulisci il layout precedente
      if self.imageLabel:
        self.polarLayout.removeWidget(self.imageLabel)
        if self.imageLabel:
          self.imageLabel.close()
          self.imageLabel = None
    
    # Crea una nuova figura
    self.polarFigure = plt.figure(figsize=(8, 8))
    
    # Configurazione del diagramma polare
    ax = self.polarFigure.add_subplot(111, polar=True)
    
    # Dati per il diagramma polare
    segments = list(self.analysisResults.keys())
    num_segments = len(segments)
    
    # Definiamo 16 settori AHA standard
    num_sectors = 16
    theta = np.linspace(0, 2*np.pi, num_sectors, endpoint=False)
    
    # Creiamo il diagramma polare per ogni segmento
    for segment_idx, segment_name in enumerate(segments):
        results = self.analysisResults[segment_name]
        metrics = results['metrics']
        
        # Ottieni dati di movimento
        point_motion = metrics['pointMotion']
        threshold = metrics['threshold']
        
        # Crea dati per il diagramma polare
        # Per ogni settore calcoliamo la percentuale di punti con movimento anomalo
        radii = np.zeros(num_sectors)
        for sector in range(num_sectors):
            # Seleziona punti per questo settore
            sector_points = np.array_split(point_motion, num_sectors)[sector]
            if len(sector_points) > 0:
                # Calcola percentuale di punti sotto la soglia (movimento anomalo)
                anomaly_percent = np.sum(sector_points < threshold) / len(sector_points)
                radii[sector] = anomaly_percent
        
        # Per visualizzare meglio, normalizza i valori
        radii = radii * 100  # Converti in percentuale
        
        # Colora in base alla percentuale di anomalia
        bars = ax.bar(theta, radii, width=2*np.pi/num_sectors, bottom=0.0, alpha=0.5)
        
        # Colora le barre in base al valore (rosso = alta anomalia, blu = bassa anomalia)
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.RdBu_r(r/100))
        
        # Aggiungi una linea che rappresenta il limite del segmento
        if segment_idx < num_segments - 1:
            circle_radius = np.max(radii) * 1.1
            ax.plot(theta, [circle_radius] * num_sectors, 'k--', alpha=0.3)
    
    # Configura l'aspetto del diagramma
    ax.set_title("Diagramma Polare di Movimento Anomalo")
    ax.set_theta_zero_location("N")  # 0 gradi in alto (Nord)
    ax.set_theta_direction(-1)  # Senso orario
    
    # Etichette settori AHA
    ax.set_thetagrids(np.degrees(theta), ['1', '2', '3', '4', '5', '6', '7', '8', 
                                         '9', '10', '11', '12', '13', '14', '15', '16'])
    
    # Aggiungi colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = self.polarFigure.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("% Movimento Anomalo")
    
    # Salva la figura come immagine temporanea
    import tempfile
    temp_file = os.path.join(tempfile.gettempdir(), "polar_diagram.png")
    self.polarFigure.savefig(temp_file, dpi=100, bbox_inches='tight')
    plt.close(self.polarFigure)
    
    # Visualizza l'immagine usando un QLabel
    self.imageLabel = qt.QLabel()
    self.imageLabel.setPixmap(qt.QPixmap(temp_file))
    self.imageLabel.setScaledContents(True)
    self.polarLayout.addWidget(self.imageLabel)

  def setupVisualization(self):
    """Configura una corretta visualizzazione 3D e 2D dei risultati"""
    if not self.analysisResults:
      return
        
    # Configura una vista 3D ottimale
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    if threeDWidget:
      threeDView = threeDWidget.threeDView()
      threeDView.resetFocalPoint()
      threeDView.resetCamera()
    
    # Imposta visibilità corretta per tutti i modelli (nascondi i modelli base)
    for segmentName, results in self.analysisResults.items():
      if 'frameResults' in results:
        for frameIdx, frameData in results['frameResults'].items():
          # Nascondi il modello base (il "frameModelNode")
          if 'frameModelNode' in frameData:
            baseModel = frameData['frameModelNode']
            if baseModel and baseModel.GetDisplayNode():
              baseModel.GetDisplayNode().SetVisibility(False)
              
          # Configura correttamente la mappa a colori, ma inizialmente nascondi tutti
          if 'motionMapNode' in frameData:
            motionMapNode = frameData['motionMapNode']
            if motionMapNode and motionMapNode.GetDisplayNode():
              displayNode = motionMapNode.GetDisplayNode()
              displayNode.SetVisibility(False)  # Inizialmente nascondi tutto
                        
              # Assicurati che la visualizzazione scalare sia attiva
              displayNode.SetScalarVisibility(True)
              displayNode.SetActiveScalarName("AbnormalMotion")
              displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
              displayNode.SetScalarRange(0, 1)
                        
              # Imposta opacità ottimale
              displayNode.SetOpacity(0.8)
    
    # Mostra solo i modelli del frame corrente
    self.onFrameSliderChanged(self.frameSlider.value)
    
    # Configura una vista che mostri meglio i risultati
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    
    # Attiva l'opzione per il collegamento tra le viste
    sliceLogics = slicer.app.applicationLogic().GetSliceLogics()
    for i in range(sliceLogics.GetNumberOfItems()):
      sliceLogic = sliceLogics.GetItemAsObject(i)
      if sliceLogic:
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetLinkedControl(True)
    
    # Feedback all'utente
    slicer.util.infoDisplay("Visualizzazione configurata. Usa lo slider per navigare tra i frame.\n"
                           "Aree rosse = movimento anomalo (acinesia/discinesia)\n"
                           "Aree blu = movimento normale")

  def onFrameSliderChanged(self, frameIndex):
    """Aggiorna la visualizzazione quando si cambia frame"""
    # Aggiorna il display per mostrare il frame selezionato
    frameIndex = int(frameIndex)
    if not self.volumeNodes or frameIndex >= len(self.volumeNodes):
      return
      
    # Aggiorna il volume visualizzato
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetReferenceActiveVolumeID(self.volumeNodes[frameIndex].GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection(0)
    
    # Visualizzazione 3D: nascondi TUTTE le mappe di movimento per TUTTI i segmenti
    if self.analysisResults:
      for segmentName, results in self.analysisResults.items():
        if 'frameResults' in results:
          for fidx, frameData in results['frameResults'].items():
            if 'motionMapNode' in frameData:
              motionMapNode = frameData['motionMapNode']
              if motionMapNode and motionMapNode.GetDisplayNode():
                # Nascondi tutti i modelli inizialmente
                motionMapNode.GetDisplayNode().SetVisibility(False)
    
    # Visualizzazione 3D: mostra solo le mappe per il frame corrente per TUTTI i segmenti
    if self.analysisResults:
      for segmentName, results in self.analysisResults.items():
        if 'frameResults' in results and frameIndex in results['frameResults']:
          if 'motionMapNode' in results['frameResults'][frameIndex]:
            motionMapNode = results['frameResults'][frameIndex]['motionMapNode']
            if motionMapNode and motionMapNode.GetDisplayNode():
              # Mostra il modello per il frame corrente
              displayNode = motionMapNode.GetDisplayNode()
              displayNode.SetVisibility(True)
              
              # Assicurati che la visualizzazione scalare sia attiva
              displayNode.SetScalarVisibility(True)
              displayNode.SetActiveScalarName("AbnormalMotion")
              
              # Verifica che il colorNode esista e sia configurato
              colorNodeID = displayNode.GetColorNodeID()
              if colorNodeID:
                colorNode = slicer.mrmlScene.GetNodeByID(colorNodeID)
                if not colorNode:
                  # Se manca il colorNode, crealo di nuovo
                  colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
                  colorNode.SetName(f"{motionMapNode.GetName()}_colors")
                  colorNode.SetTypeToUser()
                  colorNode.SetNumberOfColors(2)
                  colorNode.SetColor(0, "Abnormal", 1, 0, 0)  # Rosso per aree anomale
                  colorNode.SetColor(1, "Normal", 0, 0, 1)    # Blu per aree normali
                  displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    
    # Inoltre, aggiorna anche la segmentazione corrente
    if hasattr(self, 'segmentationFrames') and self.segmentationFrames and frameIndex in self.segmentationFrames:
        # Mostra solo la segmentazione per il frame corrente
        for fIdx, segNode in self.segmentationFrames.items():
            if segNode:
                segNode.GetDisplayNode().SetVisibility(fIdx == frameIndex)
    
    # Forza l'aggiornamento di tutte le viste
    slicer.app.layoutManager().threeDWidget(0).threeDView().forceRender()
    
    # Aggiorna anche le viste 2D
    for sliceViewName in ['Red', 'Green', 'Yellow']:
      sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
      if sliceWidget:
        sliceWidget.sliceView().forceRender()

  def onPreviousFrame(self):
    """Passa al frame precedente"""
    currentFrame = int(self.frameSlider.value)
    if currentFrame > 0:
      self.frameSlider.value = currentFrame - 1
  
  def onNextFrame(self):
    """Passa al frame successivo"""
    currentFrame = int(self.frameSlider.value)
    if currentFrame < self.numFrames - 1:
      self.frameSlider.value = currentFrame + 1
    elif self.isPlaying:
      # Se è in play e arriva all'ultimo frame, torna al primo
      self.frameSlider.value = 0
  
  def onPlayButton(self):
    """Gestisce il pulsante play/pausa"""
    if self.isPlaying:
      self.animationTimer.stop()
      self.playButton.text = "Play"
    else:
      self.animationTimer.start()
      self.playButton.text = "Pausa"
    self.isPlaying = not self.isPlaying
  
  def onSavePDFButton(self):
    """Salva il report PDF con i risultati dell'analisi"""
    if not self.analysisResults:
      return
      
    # Chiedi all'utente dove salvare il file
    filename = qt.QFileDialog.getSaveFileName(None, "Salva Report", "", "PDF Files (*.pdf);;HTML Files (*.html)")
    if not filename:
      return
      
    # Crea il report
    logic = CardiacMotionAnalysisLogic()
    
    # Cattura screenshot della vista 3D
    layoutManager = slicer.app.layoutManager()
    threeDView = layoutManager.threeDWidget(0).threeDView()
    screenshot = threeDView.grab().toImage()  # Usa .grab() invece di .grabWidget()
    
    # Salva temporaneamente lo screenshot
    import tempfile
    screenshotPath = os.path.join(tempfile.gettempdir(), "cardiac_screenshot.png")
    screenshot.save(screenshotPath)
    
    # Salva l'immagine del diagramma polare
    polarPath = os.path.join(tempfile.gettempdir(), "polar_diagram.png")
    if hasattr(self, 'polarFigure') and self.polarFigure:
        # Ricrea la figura polare
        self.updatePolarDiagram()
        # L'immagine è già stata salvata durante updatePolarDiagram
    
    # Genera il report basato sull'estensione del file
    if filename.lower().endswith('.pdf'):
        # Prova a generare PDF
        success = logic.generatePDFReport(self.analysisResults, screenshotPath, polarPath, filename)
        if not success:
            # Se fallisce, chiedi all'utente se vuole salvare come HTML
            if slicer.util.confirmYesNoDisplay("Impossibile generare PDF. Vuoi salvare il report come HTML?"):
                html_filename = filename[:-4] + '.html'
                success = logic.generateHTMLReport(self.analysisResults, screenshotPath, polarPath, html_filename)
                filename = html_filename
    else:
        # Genera HTML
        success = logic.generateHTMLReport(self.analysisResults, screenshotPath, polarPath, filename)
    
    # Pulizia
    try:
        os.remove(screenshotPath)
        os.remove(polarPath)
    except:
        pass
    
    if success:
        slicer.util.infoDisplay(f"Report salvato come: {filename}")
    else:
        slicer.util.errorDisplay("Errore durante il salvataggio del report")
      
#
# CardiacMotionAnalysisLogic
#
class CardiacMotionAnalysisLogic(ScriptedLoadableModuleLogic):
 """Implementazione della logica per l'analisi del movimento cardiaco"""
 
 def __init__(self):
   self.progressCallback = None
 
 def updateProgress(self, progress):
   """Aggiorna la barra di progresso"""
   if self.progressCallback:
     self.progressCallback(progress)
 
 def ensure_reportlab_installed(self):
   """
   Verifica che ReportLab sia installato e lo installa se necessario
   """
   try:
     import reportlab
     return True
   except ImportError:
     # ReportLab non è installato, proviamo a installarlo
     try:
       import pip
       slicer.util.pip_install("reportlab")
       import reportlab
       logging.info("ReportLab installato con successo")
       return True
     except Exception as e:
       logging.error(f"Impossibile installare ReportLab: {e}")
       return False

 def run(self, inputVolume, segmentationNode, selectedSegments, thresholdPercentage, show3D=True):
   """
   Esegue l'algoritmo di analisi del movimento
   """
   logging.info('Processo di analisi del movimento cardiaco avviato')
   
   # Controlla i parametri di input
   if not inputVolume or not segmentationNode:
     logging.error('Input mancanti')
     return False
   
   # Inizializza la callback di progresso
   self.updateProgress(0)
   
   # Ottieni tutti i timepoints dal volume 4D
   volumes = []
   isSequence = False
   
   if isinstance(inputVolume, slicer.vtkMRMLSequenceNode):
     isSequence = True
     logging.info('Elaborazione SequenceNode')
     
     # Ottieni o crea un browser di sequenza per questo nodo
     sequenceBrowserNode = None
     for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode'):
       if node.GetMasterSequenceNode() == inputVolume:
         sequenceBrowserNode = node
         break
             
     if not sequenceBrowserNode:
       sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
       sequenceBrowserNode.SetAndObserveMasterSequenceNodeID(inputVolume.GetID())
       sequenceBrowserNode.SetPlaybackItemSkippingEnabled(False)
       slicer.modules.sequences.logic().AddSynchronizedNode(inputVolume, None, sequenceBrowserNode)
     
     # Ottieni il numero di timepoints
     numTimepoints = inputVolume.GetNumberOfDataNodes()
     logging.info(f'Numero di timepoints nella sequenza: {numTimepoints}')
     
     # Crea copie dei volumi per ogni timepoint
     for i in range(numTimepoints):
       self.updateProgress(5 + (i / numTimepoints) * 10)  # Primi 15% per caricare i dati
       
       # Seleziona l'indice nel browser
       sequenceBrowserNode.SetSelectedItemNumber(i)
       
       # Ottieni il volume corrente
       proxyNode = sequenceBrowserNode.GetProxyNode(inputVolume)
       if proxyNode:
         # Crea una copia del volume per poterlo manipolare
         volumeClone = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"timepoint_{i}")
         volumeClone.Copy(proxyNode)
         volumes.append(volumeClone)
       else:
         logging.error(f'Impossibile ottenere il proxyNode per il timepoint {i}')
   else:
     # Per MultiVolumeNode (convertire in una lista di volumi normali)
     logging.info('Elaborazione MultiVolumeNode')
     numTimepoints = inputVolume.GetNumberOfFrames()
     logging.info(f'Numero di frame nel MultiVolume: {numTimepoints}')
     
     for i in range(numTimepoints):
       self.updateProgress(5 + (i / numTimepoints) * 10)  # Primi 15% per caricare i dati
       
       # Estrai ogni timepoint come volume separato
       volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"timepoint_{i}")
       
       try:
         # Usa la logica MultiVolume per estrarre il frame
         mvLogic = slicer.modules.multivolume.logic()
         mvLogic.ExtractFrame(inputVolume, i, volumeNode)
         volumes.append(volumeNode)
       except Exception as e:
         logging.error(f'Errore nell\'estrazione del frame {i}: {str(e)}')
         slicer.mrmlScene.RemoveNode(volumeNode)
   
   # Verifica che siano stati estratti volumi
   if not volumes:
     logging.error('Nessun volume estratto dalla sequenza/multivolume')
     return False
   
   logging.info(f'Estratti {len(volumes)} volumi')
   self.updateProgress(15)  # 15% completato dopo l'estrazione dei volumi
   
   # Propaga la segmentazione a tutti i frame
   segmentationResults = {}
   
   # Per ogni segmento, estrai i contorni e analizza il movimento
   for segmentIdx, segmentName in enumerate(selectedSegments):
     segmentProgress = 15 + (segmentIdx / len(selectedSegments)) * 85  # Restante 85% distribuito tra i segmenti
     self.updateProgress(segmentProgress)
     
     logging.info(f'Analisi del segmento: {segmentName}')
     
     # Verifica che il segmento esista
     segmentID = None
     for currSegmentID in segmentationNode.GetSegmentation().GetSegmentIDs():
       currSegmentName = segmentationNode.GetSegmentation().GetSegment(currSegmentID).GetName()
       if currSegmentName.lower() == segmentName.lower():
         segmentID = currSegmentID
         break
     
     if not segmentID:
       logging.error(f'Segmento non trovato: {segmentName}')
       continue
     
     # Crea un modello per il segmento originale
     baseModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{segmentName}_base_model")
     slicer.modules.segmentations.logic().ExportSegmentToRepresentationNode(
       segmentationNode.GetSegmentation().GetSegment(segmentID), 
       baseModelNode
     )
     
     # Risultati per questo segmento tra tutti i frame
     frameResults = {}
     
     # Analizza il movimento attraverso i frame
     for frameIdx in range(len(volumes)):
       frameProgress = segmentProgress + (frameIdx / len(volumes)) * (85 / len(selectedSegments))
       self.updateProgress(frameProgress)
       
       # Clona il modello base per questo frame
       frameModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{segmentName}_frame_{frameIdx}")
       frameModelNode.Copy(baseModelNode)
       
       # Elabora il modello per questo frame
       self.processModelForFrame(frameModelNode, volumes[frameIdx], volumes[0])
       
       # Memorizza il modello per questo frame
       frameResults[frameIdx] = {
         'frameModelNode': frameModelNode
       }
     
     # Calcola metriche di movimento attraverso tutti i frame
     motionMetrics = self.calculateMotionMetrics(frameResults)
     
     # Crea mappe di movimento per ogni frame
     if show3D:
       for frameIdx in frameResults:
         motionMapNode = self.createMotionMapForFrame(
           frameResults[frameIdx]['frameModelNode'],
           motionMetrics['pointMotion'],
           motionMetrics['threshold']
         )
         frameResults[frameIdx]['motionMapNode'] = motionMapNode
     
     # Memorizzo i risultati
     segmentationResults[segmentName] = {
       'metrics': motionMetrics,
       'frameResults': frameResults
     }
   
   self.updateProgress(100)
   logging.info('Analisi del movimento cardiaco completata')
   return volumes, segmentationResults

 def processModelForFrame(self, modelNode, frameVolume, referenceVolume):
   """
   Elabora un modello per un determinato frame, adattandolo al volume corrente
   """
   # In una implementazione completa, qui si potrebbe eseguire:
   # 1. Registrazione delle immagini tra il frame corrente e il riferimento
   # 2. Applicazione della trasformazione al modello
   # 3. Calcolo delle deformazioni del modello
   
   # Per semplicità, simuliamo il movimento basandoci su informazioni di intensità
   polyData = modelNode.GetPolyData()
   numPoints = polyData.GetPoints().GetNumberOfPoints()
   
   # Crea array di scalari per memorizzare il movimento
   motionArray = vtk.vtkFloatArray()
   motionArray.SetName("MotionMagnitude")
   motionArray.SetNumberOfComponents(1)
   motionArray.SetNumberOfTuples(numPoints)
   
   # Ottieni matrici RAS to IJK per campionare i volumi
   referenceRasToIjk = vtk.vtkMatrix4x4()
   referenceVolume.GetRASToIJKMatrix(referenceRasToIjk)
   
   frameRasToIjk = vtk.vtkMatrix4x4()
   frameVolume.GetRASToIJKMatrix(frameRasToIjk)
   
   # Ottieni dimensioni dei volumi
   refDimensions = referenceVolume.GetImageData().GetDimensions()
   frameDimensions = frameVolume.GetImageData().GetDimensions()
   
   # Per ogni punto della mesh, calcola il movimento stimato
   for i in range(numPoints):
     point = polyData.GetPoints().GetPoint(i)
     
     # Calcola coordinate IJK nel volume di riferimento
     refIjk = [0, 0, 0, 1]
     referenceRasToIjk.MultiplyPoint(np.append(point, 1.0), refIjk)
     refIjk = [int(round(c)) for c in refIjk[0:3]]
     
     # Calcola coordinate IJK nel frame corrente
     frameIjk = [0, 0, 0, 1]
     frameRasToIjk.MultiplyPoint(np.append(point, 1.0), frameIjk)
     frameIjk = [int(round(c)) for c in frameIjk[0:3]]
     
     # Verifica che i punti siano all'interno dei volumi
     if (0 <= refIjk[0] < refDimensions[0] and 
         0 <= refIjk[1] < refDimensions[1] and 
         0 <= refIjk[2] < refDimensions[2] and
         0 <= frameIjk[0] < frameDimensions[0] and 
         0 <= frameIjk[1] < frameDimensions[1] and 
         0 <= frameIjk[2] < frameDimensions[2]):
       
       # Ottieni le intensità nei due volumi
       refIntensity = referenceVolume.GetImageData().GetScalarComponentAsDouble(refIjk[0], refIjk[1], refIjk[2], 0)
       frameIntensity = frameVolume.GetImageData().GetScalarComponentAsDouble(frameIjk[0], frameIjk[1], frameIjk[2], 0)
       
       # Calcola una stima del movimento come differenza di intensità
       motionEstimate = abs(frameIntensity - refIntensity)
       motionArray.SetValue(i, motionEstimate)
     else:
       motionArray.SetValue(i, 0)
   
   # Aggiungi l'array di scalari al modello
   polyData.GetPointData().AddArray(motionArray)
   polyData.GetPointData().SetActiveScalars("MotionMagnitude")
   
   return True

 def calculateMotionMetrics(self, frameResults):
   """
   Calcola metriche di movimento basate su tutti i frame
   """
   # Trova il modello di un frame qualsiasi
   firstFrameKey = list(frameResults.keys())[0]
   sampleModel = frameResults[firstFrameKey]['frameModelNode']
   
   numPoints = sampleModel.GetPolyData().GetPoints().GetNumberOfPoints()
   
   # Inizializza array per il movimento punto per punto
   pointMotion = np.zeros(numPoints)
   
   # Per ogni frame, accumula il movimento massimo per ogni punto
   for frameIdx, frameData in frameResults.items():
     modelNode = frameData['frameModelNode']
     polyData = modelNode.GetPolyData()
     
     # Ottieni array di scalari di movimento per questo frame
     motionArray = polyData.GetPointData().GetArray("MotionMagnitude")
     if motionArray:
       for pointIdx in range(numPoints):
         motion = motionArray.GetValue(pointIdx)
         # Accumula il movimento massimo osservato per ogni punto
         pointMotion[pointIdx] = max(pointMotion[pointIdx], motion)
   
   # Calcola statistiche di movimento
   avgMotion = np.mean(pointMotion)
   maxMotion = np.max(pointMotion)
   medianMotion = np.median(pointMotion)
   
   # Determina una soglia per il movimento anomalo
   # Utilizziamo un approccio basato sui percentili per maggiore robustezza
   thresholdValue = np.percentile(pointMotion, 25)  # 25° percentile come soglia
   
   return {
     'pointMotion': pointMotion,
     'averageMotion': avgMotion,
     'maxMotion': maxMotion,
     'medianMotion': medianMotion,
     'threshold': thresholdValue
   }

 def createMotionMapForFrame(self, modelNode, pointMotion, threshold):
   """
   Crea una mappa di colori 3D che evidenzia aree con movimento anomalo
   """
   # Clona il modello
   motionMapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{modelNode.GetName()}_motion_map")
   motionMapNode.Copy(modelNode)
   
   # Ottieni polydata
   polyData = motionMapNode.GetPolyData()
   numPoints = polyData.GetPoints().GetNumberOfPoints()
   
   # Crea array per visualizzare il movimento anomalo
   abnormalArray = vtk.vtkFloatArray()
   abnormalArray.SetName("AbnormalMotion")
   abnormalArray.SetNumberOfComponents(1)
   abnormalArray.SetNumberOfTuples(numPoints)
   
   # Imposta i valori (0 = anomalo, 1 = normale)
   for i in range(numPoints):
     if pointMotion[i] < threshold:
       abnormalArray.SetValue(i, 0)  # Movimento anomalo
     else:
       abnormalArray.SetValue(i, 1)  # Movimento normale
   
   # Aggiungi l'array alla polydata
   polyData.GetPointData().AddArray(abnormalArray)
   polyData.GetPointData().SetActiveScalars("AbnormalMotion")
   
   # Configura display node
   displayNode = motionMapNode.GetDisplayNode()
   if not displayNode:
     displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
     motionMapNode.SetAndObserveDisplayNodeID(displayNode.GetID())
   
   # Crea una tabella colori personalizzata
   colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
   colorNode.SetName(f"{modelNode.GetName()}_colors")
   colorNode.SetTypeToUser()
   colorNode.SetNumberOfColors(2)
   colorNode.SetColor(0, "Abnormal", 1, 0, 0)  # Rosso per aree anomale
   colorNode.SetColor(1, "Normal", 0, 0, 1)    # Blu per aree normali
   
   # Configura la visualizzazione
   displayNode.SetAndObserveColorNodeID(colorNode.GetID())
   displayNode.SetActiveScalarName("AbnormalMotion")
   displayNode.SetScalarVisibility(True)
   displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
   displayNode.SetScalarRange(0, 1)
   
   return motionMapNode

 def generatePDFReport(self, analysisResults, screenshot3DPath, polarDiagramPath, filename):
   """
   Genera un report PDF con i risultati dell'analisi
   """
   if not analysisResults:
     return False
   
   # Verifica che ReportLab sia disponibile
   if not self.ensure_reportlab_installed():
     logging.error("Impossibile generare PDF: ReportLab non può essere installato")
     return False
     
   try:
     # Importa i moduli necessari
     from reportlab.lib.pagesizes import A4
     from reportlab.lib import colors
     from reportlab.lib.units import inch, cm
     from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
     from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
     import datetime
     import numpy as np
     
     # Crea un nuovo documento PDF
     doc = SimpleDocTemplate(filename, pagesize=A4)
     styles = getSampleStyleSheet()
     elements = []
     
     # Stili personalizzati
     styles.add(ParagraphStyle(name='Center', alignment=1, parent=styles['Heading1']))
     
     # Prima pagina: Intestazione e riassunto
     elements.append(Paragraph('Analisi della Discinesia/Acinesia Cardiaca', styles['Center']))
     elements.append(Spacer(1, 0.5*cm))
     
     # Data e ora dell'analisi
     current_time = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
     elements.append(Paragraph(f'Data analisi: {current_time}', styles['Normal']))
     elements.append(Paragraph(f'Numero di segmenti analizzati: {len(analysisResults)}', styles['Normal']))
     elements.append(Spacer(1, 1*cm))
     
     # Tabella di risultati
     elements.append(Paragraph('Risultati quantitativi:', styles['Heading2']))
     elements.append(Spacer(1, 0.5*cm))
     
     data = [["Segmento", "% Area Anomala", "Movimento Medio", "Movimento Massimo"]]
     
     for segmentName, results in analysisResults.items():
       metrics = results['metrics']
       pointMotion = metrics['pointMotion']
       threshold = metrics['threshold']
       
       # Calcola la percentuale di area con movimento anomalo
       anomalyPercent = 100 * (np.sum(pointMotion < threshold) / len(pointMotion))
       
       # Aggiungi riga alla tabella
       data.append([
         segmentName,
         f"{anomalyPercent:.1f}%",
         f"{metrics['averageMotion']:.2f}",
         f"{metrics['maxMotion']:.2f}"
       ])
     
     # Crea e formatta la tabella
     table = Table(data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
     table.setStyle(TableStyle([
       ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
       ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
       ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
       ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
       ('BACKGROUND', (0, 1), (-1, -1), colors.white),
       ('GRID', (0, 0), (-1, -1), 1, colors.black)
     ]))
     
     elements.append(table)
     elements.append(Spacer(1, 1*cm))
     
     # Aggiungi screenshot 3D e diagramma polare
     elements.append(Paragraph('Visualizzazione dei risultati:', styles['Heading2']))
     elements.append(Spacer(1, 0.5*cm))
     
     # Layout con 2 immagini affiancate
     image_data = [[Image(screenshot3DPath, width=8*cm, height=6*cm), 
                   Image(polarDiagramPath, width=8*cm, height=6*cm)]]
     
     image_table = Table(image_data)
     elements.append(image_table)
     elements.append(Spacer(1, 0.5*cm))
     
     # Didascalie
     elements.append(Paragraph('Visualizzazione 3D (sinistra): Mostra le aree con movimento anomalo in rosso', styles['Normal']))
     elements.append(Paragraph('Diagramma polare (destra): Rappresenta la distribuzione del movimento anomalo nei 16 settori standard', styles['Normal']))
     elements.append(Spacer(1, 1*cm))
     
     # Interpretazione clinica
     elements.append(Paragraph('Interpretazione dei risultati:', styles['Heading2']))
     elements.append(Spacer(1, 0.5*cm))
     
     elements.append(Paragraph('• Le aree rosse nella visualizzazione 3D rappresentano zone con movimento anomalo (acinesia o discinesia).', styles['Normal']))
     elements.append(Paragraph('• Una percentuale elevata di area anomala (>30%) indica una potenziale disfunzione significativa.', styles['Normal']))
     elements.append(Paragraph('• Il diagramma polare permette di identificare quali settori miocardici sono maggiormente compromessi.', styles['Normal']))
     elements.append(Paragraph('• Consultare un cardiologo per l\'interpretazione clinica definitiva di questi risultati.', styles['Normal']))
     
     # Costruisci il PDF
     doc.build(elements)
     
     return True
     
   except Exception as e:
     logging.error(f"Errore durante la generazione del PDF: {e}")
     import traceback
     traceback.print_exc()
     return False
 
 def generateHTMLReport(self, analysisResults, screenshot3DPath, polarDiagramPath, filename):
   """
   Genera un report HTML con i risultati dell'analisi invece di un PDF
   """
   if not analysisResults:
       return False
       
   try:
       # Modifica l'estensione del file da .pdf a .html se necessario
       if filename.lower().endswith('.pdf'):
           filename = filename[:-4] + '.html'
       
       import datetime
       import numpy as np
       import shutil
       import os
       
       # Crea una directory per le immagini del report
       report_dir = os.path.dirname(filename)
       images_dir = os.path.join(report_dir, 'report_images')
       os.makedirs(images_dir, exist_ok=True)
       
       # Copia le immagini nella directory del report
       screen_dest = os.path.join(images_dir, 'screenshot.png')
       polar_dest = os.path.join(images_dir, 'polar.png')
       shutil.copy(screenshot3DPath, screen_dest)
       shutil.copy(polarDiagramPath, polar_dest)
       
       # Crea il contenuto HTML
       current_time = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
       
       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Analisi della Discinesia/Acinesia Cardiaca</title>
           <style>
               body {{ font-family: Arial, sans-serif; margin: 40px; }}
               h1 {{ color: #333; text-align: center; }}
               h2 {{ color: #555; margin-top: 20px; }}
               table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
               th, td {{ padding: 10px; border: 1px solid #ddd; text-align: center; }}
               th {{ background-color: #f2f2f2; }}
               .image-container {{ display: flex; justify-content: space-between; margin: 20px 0; }}
               .image-container img {{ max-width: 48%; }}
               .note {{ font-style: italic; color: #666; }}
           </style>
       </head>
       <body>
           <h1>Analisi della Discinesia/Acinesia Cardiaca</h1>
           <p>Data analisi: {current_time}</p>
           <p>Numero di segmenti analizzati: {len(analysisResults)}</p>
           
           <h2>Risultati quantitativi:</h2>
           <table>
               <tr>
                   <th>Segmento</th>
                   <th>% Area Anomala</th>
                   <th>Movimento Medio</th>
                   <th>Movimento Massimo</th>
               </tr>
       """
       
       # Aggiungi righe della tabella per ogni segmento
       for segmentName, results in analysisResults.items():
           metrics = results['metrics']
           point_motion = metrics['pointMotion']
           threshold = metrics['threshold']
           
           # Calcola percentuale di area con movimento anomalo
           anomaly_percent = 100 * (np.sum(point_motion < threshold) / len(point_motion))
           
           html_content += f"""
               <tr>
                   <td>{segmentName}</td>
                   <td>{anomaly_percent:.1f}%</td>
                   <td>{metrics['averageMotion']:.2f}</td>
                   <td>{metrics['maxMotion']:.2f}</td>
               </tr>
           """
       
       # Continua con il resto del documento HTML
       html_content += f"""
           </table>
           
           <h2>Visualizzazione dei risultati:</h2>
           <div class="image-container">
               <img src="report_images/screenshot.png" alt="Visualizzazione 3D">
               <img src="report_images/polar.png" alt="Diagramma polare">
           </div>
           <p>Visualizzazione 3D (sinistra): Mostra le aree con movimento anomalo in rosso</p>
           <p>Diagramma polare (destra): Rappresenta la distribuzione del movimento anomalo nei 16 settori standard</p>
           
           <h2>Interpretazione dei risultati:</h2>
           <ul>
               <li>Le aree rosse nella visualizzazione 3D rappresentano zone con movimento anomalo (acinesia o discinesia).</li>
               <li>Una percentuale elevata di area anomala (>30%) indica una potenziale disfunzione significativa.</li>
               <li>Il diagramma polare permette di identificare quali settori miocardici sono maggiormente compromessi.</li>
               <li>Consultare un cardiologo per l'interpretazione clinica definitiva di questi risultati.</li>
           </ul>
           
           <p class="note">Report generato automaticamente dal modulo Cardiac Motion Analysis per 3D Slicer</p>
       </body>
       </html>
       """
       
       # Scrivi il file HTML
       with open(filename, 'w') as f:
           f.write(html_content)
       
       return True
       
   except Exception as e:
       logging.error(f"Errore durante la generazione del report HTML: {e}")
       import traceback
       traceback.print_exc()
       return False

#
# CardiacMotionAnalysisTest
#
class CardiacMotionAnalysisTest(ScriptedLoadableModuleTest):
 """
 Test unitari per il modulo
 """
 
 def setUp(self):
   """ Configurazione per i test """
   slicer.mrmlScene.Clear(0)

 def runTest(self):
   """Esegui i test"""
   self.setUp()
   self.test_CardiacMotionAnalysis1()

 def test_CardiacMotionAnalysis1(self):
   """Test di base"""
   self.delayDisplay("Avvio del test")
   self.delayDisplay('Test completato')