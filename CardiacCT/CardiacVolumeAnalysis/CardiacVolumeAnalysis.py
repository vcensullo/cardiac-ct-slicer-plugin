import os
import time
import numpy as np
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import vtkSegmentationCorePython as vtkSegmentationCore

class CardiacVolumeAnalysis(ScriptedLoadableModule):
  """Modulo per l'analisi dei volumi cardiaci da segmentazioni esistenti"""
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Cardiac Volume Analysis"
    self.parent.categories = ["CardiacCT"]
    self.parent.dependencies = []
    self.parent.contributors = ["Vittorio Censullo, AITeRTC"]
    self.parent.helpText = """
    Questo modulo analizza segmentazioni cardiache esistenti generate con TotalSegmentator.
    """
    self.parent.acknowledgementText = """
    Questo modulo analizza segmentazioni create con TotalSegmentator.
    """

class CardiacVolumeAnalysisWidget(ScriptedLoadableModuleWidget):
  """Interfaccia utente per il modulo"""

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    
    # Parametri
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parametri"
    self.layout.addWidget(parametersCollapsibleButton)
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    
    # Segmentazione
    self.segmentationSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segmentationSelector.selectNodeUponCreation = True
    self.segmentationSelector.addEnabled = False
    self.segmentationSelector.removeEnabled = False
    self.segmentationSelector.noneEnabled = False
    self.segmentationSelector.showHidden = False
    self.segmentationSelector.showChildNodeTypes = False
    self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentationSelector.setToolTip("Seleziona la segmentazione cardiaca")
    parametersFormLayout.addRow("Segmentazione: ", self.segmentationSelector)
    
    # Volume Sequence
    self.volumeSequenceSelector = slicer.qMRMLNodeComboBox()
    self.volumeSequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.volumeSequenceSelector.selectNodeUponCreation = True
    self.volumeSequenceSelector.addEnabled = False
    self.volumeSequenceSelector.removeEnabled = False
    self.volumeSequenceSelector.noneEnabled = True
    self.volumeSequenceSelector.showHidden = False
    self.volumeSequenceSelector.showChildNodeTypes = False
    self.volumeSequenceSelector.setMRMLScene(slicer.mrmlScene)
    self.volumeSequenceSelector.setToolTip("Seleziona la sequenza di volume (opzionale)")
    parametersFormLayout.addRow("Sequenza di volume: ", self.volumeSequenceSelector)
    
    # Numero di fasi
    self.numPhasesSpinBox = qt.QSpinBox()
    self.numPhasesSpinBox.minimum = 1
    self.numPhasesSpinBox.maximum = 100
    self.numPhasesSpinBox.value = 20  # Valore predefinito temporaneo
    self.numPhasesSpinBox.setToolTip("Numero di fasi temporali nel ciclo cardiaco")
    parametersFormLayout.addRow("Numero di fasi: ", self.numPhasesSpinBox)
    
    # Opzioni di analisi
    optionsCollapsibleButton = ctk.ctkCollapsibleButton()
    optionsCollapsibleButton.text = "Opzioni di analisi"
    self.layout.addWidget(optionsCollapsibleButton)
    optionsFormLayout = qt.QFormLayout(optionsCollapsibleButton)
    
    # Cartella di output
    self.outputDirSelector = ctk.ctkPathLineEdit()
    self.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirSelector.setToolTip("Seleziona la cartella di output")
    optionsFormLayout.addRow("Cartella di output: ", self.outputDirSelector)
    
    # Campo per il nome del paziente
    self.patientNameEdit = qt.QLineEdit()
    self.patientNameEdit.setToolTip("Inserisci il nome del paziente")
    self.patientNameEdit.setPlaceholderText("Nome del paziente")
    optionsFormLayout.addRow("Nome paziente:", self.patientNameEdit)
    
    # Checkbox per il tipo di sequenza
    self.useVolumeSequenceCheckBox = qt.QCheckBox()
    self.useVolumeSequenceCheckBox.checked = True
    self.useVolumeSequenceCheckBox.setToolTip("Attiva per utilizzare una volume sequence invece di un multivolume")
    optionsFormLayout.addRow("Usa Volume Sequence: ", self.useVolumeSequenceCheckBox)
    
    # Checkbox per creare segmentazione sincronizzata
    self.createSyncSegmentationCheckBox = qt.QCheckBox()
    self.createSyncSegmentationCheckBox.checked = False
    self.createSyncSegmentationCheckBox.setToolTip("Crea automaticamente una segmentazione sincronizzata con la sequenza")
    self.createSyncSegmentationCheckBox.enabled = True  # Inizialmente abilitata se useVolumeSequence è attiva
    optionsFormLayout.addRow("Crea segmentazione sincronizzata:", self.createSyncSegmentationCheckBox)
    
    # Pulsante di debug
    self.debugButton = qt.QPushButton("Debug Segmentazione")
    self.debugButton.toolTip = "Stampa informazioni di debug sulla segmentazione"
    self.debugButton.enabled = True
    optionsFormLayout.addRow(self.debugButton)
    
    # Pulsante per calcolare le metriche
    self.calculateButton = qt.QPushButton("Calcola metriche cardiache")
    self.calculateButton.toolTip = "Calcola volumi, massa e stroke volume per tutte le fasi"
    self.calculateButton.enabled = True
    optionsFormLayout.addRow(self.calculateButton)
    
    # Pannello di mappatura dei nomi dei segmenti
    segmentMappingCollapsibleButton = ctk.ctkCollapsibleButton()
    segmentMappingCollapsibleButton.text = "Impostazioni segmentazione"
    segmentMappingCollapsibleButton.collapsed = True  # Inizialmente collassato
    self.layout.addWidget(segmentMappingCollapsibleButton)
    segmentMappingFormLayout = qt.QFormLayout(segmentMappingCollapsibleButton)
    
    # Aggiungi campi per la mappatura dei nomi dei segmenti
    self.rightVentricleNameEdit = qt.QLineEdit("right ventricle of heart")
    self.leftVentricleNameEdit = qt.QLineEdit("left ventricle of heart")
    self.myocardiumNameEdit = qt.QLineEdit("myocardium")
    
    segmentMappingFormLayout.addRow("Nome segmento ventricolo destro:", self.rightVentricleNameEdit)
    segmentMappingFormLayout.addRow("Nome segmento ventricolo sinistro:", self.leftVentricleNameEdit)
    segmentMappingFormLayout.addRow("Nome segmento miocardio:", self.myocardiumNameEdit)
    
    # Pulsante per recuperare i nomi dalla segmentazione corrente
    self.detectSegmentNamesButton = qt.QPushButton("Rileva nomi dalla segmentazione corrente")
    self.detectSegmentNamesButton.toolTip = "Recupera i nomi dei segmenti dalla segmentazione selezionata"
    self.detectSegmentNamesButton.enabled = True
    segmentMappingFormLayout.addRow(self.detectSegmentNamesButton)
    
    # Aggiungi controlli per la selezione delle fasi cardiache
    phaseSelectionCollapsibleButton = ctk.ctkCollapsibleButton()
    phaseSelectionCollapsibleButton.text = "Selezione fasi cardiache"
    self.layout.addWidget(phaseSelectionCollapsibleButton)
    phaseSelectionFormLayout = qt.QFormLayout(phaseSelectionCollapsibleButton)
    
    # Metodo di rilevamento delle fasi
    self.phaseDetectionMethodSelector = qt.QComboBox()
    self.phaseDetectionMethodSelector.addItem("Automatico (Picchi di volume)", "auto")
    self.phaseDetectionMethodSelector.addItem("Automatico (Algoritmo avanzato)", "advanced")
    self.phaseDetectionMethodSelector.addItem("Selezione manuale", "manual")
    self.phaseDetectionMethodSelector.setToolTip("Seleziona il metodo per identificare telediastole e telesistole")
    phaseSelectionFormLayout.addRow("Metodo di rilevamento fasi: ", self.phaseDetectionMethodSelector)
    
    # Combo box per la selezione della fase di telediastole
    self.edvPhaseSelector = qt.QComboBox()
    self.edvPhaseSelector.setToolTip("Seleziona la fase di telediastole (EDV)")
    phaseSelectionFormLayout.addRow("Fase telediastole (EDV): ", self.edvPhaseSelector)
    
    # Combo box per la selezione della fase di telesistole
    self.esvPhaseSelector = qt.QComboBox()
    self.esvPhaseSelector.setToolTip("Seleziona la fase di telesistole (ESV)")
    phaseSelectionFormLayout.addRow("Fase telesistole (ESV): ", self.esvPhaseSelector)
    
    # Pulsante per aggiornare la tabella dei risultati
    self.updateResultsButton = qt.QPushButton("Aggiorna risultati")
    self.updateResultsButton.toolTip = "Aggiorna i risultati con le fasi selezionate"
    self.updateResultsButton.enabled = False
    phaseSelectionFormLayout.addRow(self.updateResultsButton)
    
    # Pulsante per esportare il PDF
    self.exportButton = qt.QPushButton("Esporta PDF")
    self.exportButton.toolTip = "Esporta i risultati in PDF"
    self.exportButton.enabled = False
    optionsFormLayout.addRow(self.exportButton)
    
    # Aggiungi tabella dei risultati
    self.resultsCollapsibleButton = ctk.ctkCollapsibleButton()
    self.resultsCollapsibleButton.text = "Risultati"
    self.layout.addWidget(self.resultsCollapsibleButton)
    resultsFormLayout = qt.QFormLayout(self.resultsCollapsibleButton)
    
    self.resultsTable = qt.QTableWidget()
    self.resultsTable.setColumnCount(5)
    self.resultsTable.setHorizontalHeaderLabels(["Fase", "Vol. Ventricolo Dx (ml)", "Vol. Ventricolo Sx (ml)", "Massa Miocardica (g)", "Stroke Volume (ml)"])
    resultsFormLayout.addRow(self.resultsTable)
    
    # Connessioni
    self.debugButton.connect('clicked(bool)', self.onDebugButton)
    self.calculateButton.connect('clicked(bool)', self.onCalculateButton)
    self.updateResultsButton.connect('clicked(bool)', self.onUpdateResults)
    self.exportButton.connect('clicked(bool)', self.onExportButton)
    self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtonStates)
    self.volumeSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtonStates)
    self.volumeSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updatePhaseCount)
    self.outputDirSelector.connect("validInputChanged(bool)", self.updateButtonStates)
    self.edvPhaseSelector.currentIndexChanged.connect(self.onUpdateResults)
    self.esvPhaseSelector.currentIndexChanged.connect(self.onUpdateResults)
    self.phaseDetectionMethodSelector.currentIndexChanged.connect(self.onPhaseDetectionMethodChanged)
    self.useVolumeSequenceCheckBox.connect("toggled(bool)", self.onUseVolumeSequenceToggled)
    self.detectSegmentNamesButton.connect('clicked(bool)', self.onDetectSegmentNames)
    
    # Inizializza variabili della classe
    self.volumeData = {}
    self.logic = CardiacVolumeAnalysisLogic()
  
  def updatePhaseCount(self):
    """Aggiorna il conteggio delle fasi in base alla sequenza selezionata"""
    sequenceNode = self.volumeSequenceSelector.currentNode()
    if sequenceNode:
        # Ottieni il numero di elementi nella sequenza
        numItems = sequenceNode.GetNumberOfDataNodes()
        if numItems > 0:
            self.numPhasesSpinBox.value = numItems
            print(f"Rilevate {numItems} fasi nella sequenza")
  
  def onUseVolumeSequenceToggled(self, checked):
    """Gestisce il cambio tra multivolume e volume sequence"""
    self.volumeSequenceSelector.setEnabled(checked)
    self.createSyncSegmentationCheckBox.setEnabled(checked)
    
    if checked:
      # Se abilitato, mostra un suggerimento su come usare le volume sequences
      slicer.util.showStatusMessage("Seleziona una sequenza di volume temporale", 3000)
    else:
      # Se disabilitato, mostra un suggerimento sul multivolume
      slicer.util.showStatusMessage("Verrà utilizzato il multivolume corrente", 3000)
  
  def updateButtonStates(self):
    """Aggiorna lo stato dei pulsanti in base alle condizioni correnti"""
    segNode = self.segmentationSelector.currentNode()
    volumeSeqNode = self.volumeSequenceSelector.currentNode() if self.useVolumeSequenceCheckBox.checked else None
    createSyncSeg = self.createSyncSegmentationCheckBox.checked if self.useVolumeSequenceCheckBox.checked else False
    hasValidOutput = bool(self.outputDirSelector.currentPath)
    
    # Abilita il pulsante di calcolo se abbiamo una segmentazione (o creeremo una) e (se richiesto) una sequenza di volume
    canCalculate = hasValidOutput and (
        (not self.useVolumeSequenceCheckBox.checked and segNode is not None) or 
        (self.useVolumeSequenceCheckBox.checked and volumeSeqNode is not None and (segNode is not None or createSyncSeg))
    )
    
    self.calculateButton.enabled = canCalculate
    self.debugButton.enabled = segNode is not None
    self.detectSegmentNamesButton.enabled = segNode is not None
  
  def _findOrCreateBrowserForSequence(self, sequenceNode):
    """Trova o crea un browser per la sequenza specificata"""
    # Get the sequence browser module logic
    seqBrowserLogic = slicer.modules.sequences.logic()
    
    # Try to find an existing browser for this sequence
    browserNode = seqBrowserLogic.GetFirstBrowserNodeForSequenceNode(sequenceNode)
    
    # If we found a browser, use it
    if browserNode:
        return browserNode
    
    # Otherwise, create a new browser
    browserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
    browserNode.SetName(sequenceNode.GetName() + " browser")
    
    # Add the sequence to the browser (don't use direct API calls)
    seqBrowserLogic.AddSynchronizedSequenceNode(sequenceNode, browserNode)
    
    # Configure the browser
    browserNode.SetPlaybackActive(False)
    browserNode.SetPlaybackItemSkippingEnabled(False)
    
    # Create proxy nodes
    seqBrowserLogic.UpdateProxyNodesFromSequences(browserNode)
    
    return browserNode
  
  def validateSequenceAndSegmentation(self):
    """Verifica che la segmentazione e la sequenza siano compatibili"""
    if not self.useVolumeSequenceCheckBox.checked:
        return True  # La validazione non è necessaria per i multivolume
    
    segNode = self.segmentationSelector.currentNode()
    volumeSeqNode = self.volumeSequenceSelector.currentNode()
    
    if not segNode or not volumeSeqNode:
        return True  # Se uno dei due non è selezionato, non possiamo validare
    
    # Use the sequence browser logic to check relationships
    seqBrowserLogic = slicer.modules.sequences.logic()
    
    # Check if the segmentation is synced with our sequence
    segBrowserNode = seqBrowserLogic.GetFirstBrowserNodeForProxyNode(segNode)
    volBrowserNode = seqBrowserLogic.GetFirstBrowserNodeForSequenceNode(volumeSeqNode)
    
    if segBrowserNode and segBrowserNode == volBrowserNode:
        return True  # They belong to the same browser
    
    # Avvisa l'utente ma continua
    warningResult = qt.QMessageBox.warning(None, "Avviso", 
                    "La segmentazione selezionata non sembra essere associata alla sequenza di volume.\n"
                    "Questo potrebbe causare problemi nell'allineamento temporale dei dati.\n\n"
                    "Vuoi continuare comunque?",
                    qt.QMessageBox.Yes | qt.QMessageBox.No)
    return warningResult == qt.QMessageBox.Yes
  
  def _createSynchronizedSegmentation(self, sequenceNode):
    """Crea una segmentazione sincronizzata con la sequenza di volume"""
    try:
        # Ottieni la logica del browser di sequenza
        seqBrowserLogic = slicer.modules.sequences.logic()
        
        # Trova o crea un browser per questa sequenza
        browser = self._findOrCreateBrowserForSequence(sequenceNode)
        if not browser:
            print("Errore: impossibile creare un browser di sequenza")
            return None
        
        # Ottieni il nodo volume proxy
        proxyVolumeNode = None
        for i in range(browser.GetNumberOfProxyNodes()):
            proxy = browser.GetProxyNode(i)
            if proxy and proxy.IsA("vtkMRMLScalarVolumeNode"):
                proxyVolumeNode = proxy
                break
        
        if not proxyVolumeNode:
            print("Errore: nessun proxy volume trovato nel browser")
            return None
        
        # Crea una nuova segmentazione
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(f"{sequenceNode.GetName()}_Segmentation")
        
        # Imposta il reference volume
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(proxyVolumeNode)
        
        # Aggiungi i segmenti predefiniti per il cuore
        self._addCardiacSegments(segmentationNode)
        
        # Aggiungi la segmentazione al browser di sequenza
        browser.AddProxyNode(segmentationNode, sequenceNode)
        
        # Aggiorna i nodi proxy
        seqBrowserLogic.UpdateProxyNodesFromSequences(browser)
        
        return segmentationNode
    
    except Exception as e:
        print(f"Errore nella creazione della segmentazione sincronizzata: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
  
  def _addCardiacSegments(self, segmentationNode):
    """Aggiunge i segmenti predefiniti per il cuore"""
    # Ottieni l'oggetto segmentazione
    segmentation = segmentationNode.GetSegmentation()
    
    # Definisci i segmenti cardiaci con colori diversi
    segmentDefinitions = [
        {"name": "right ventricle of heart", "color": [0.0, 0.0, 1.0]},  # Blu
        {"name": "left ventricle of heart", "color": [1.0, 0.0, 0.0]},   # Rosso
        {"name": "myocardium", "color": [0.0, 1.0, 0.0]}                 # Verde
    ]
    
    # Aggiungi ogni segmento
    for segDef in segmentDefinitions:
        segment = vtkSegmentationCore.vtkSegment()
        segment.SetName(segDef["name"])
        segment.SetColor(segDef["color"])
        segmentation.AddSegment(segment)
    
    # Mostra un messaggio all'utente
    slicer.util.showStatusMessage(f"Segmentazione con segmenti cardiaci creata. Usa l'Editor di Segmentazione per modificarli.", 5000)

  def onDetectSegmentNames(self):
    """Rileva i nomi dei segmenti dalla segmentazione corrente"""
    segNode = self.segmentationSelector.currentNode()
    if not segNode:
        qt.QMessageBox.warning(None, "Avviso", "Nessuna segmentazione selezionata.")
        return
    
    # Ottieni i nomi dei segmenti disponibili
    segmentation = segNode.GetSegmentation()
    segmentIDs = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(segmentIDs)
    
    segmentNames = []
    for i in range(segmentIDs.GetNumberOfValues()):
        segmentID = segmentIDs.GetValue(i)
        segment = segmentation.GetSegment(segmentID)
        segmentNames.append(segment.GetName())
    
    if not segmentNames:
        qt.QMessageBox.warning(None, "Avviso", "Nessun segmento trovato nella segmentazione.")
        return
    
    # Crea un dialogo per selezionare i segmenti
    dialog = qt.QDialog()
    dialog.setWindowTitle("Seleziona i segmenti")
    dialog.setMinimumWidth(400)
    layout = qt.QVBoxLayout(dialog)
    
    # Etichetta informativa
    label = qt.QLabel("Seleziona quale segmento corrisponde a ciascuna struttura cardiaca:")
    layout.addWidget(label)
    
    # Combo box per il ventricolo destro
    rvLabel = qt.QLabel("Ventricolo destro:")
    rvCombo = qt.QComboBox()
    rvCombo.addItems(segmentNames)
    # Trova il miglior match per RV
    best_rv_match = self._findBestMatch(segmentNames, ["right", "rv", "ventricle"])
    if best_rv_match >= 0:
        rvCombo.setCurrentIndex(best_rv_match)
    
    # Combo box per il ventricolo sinistro
    lvLabel = qt.QLabel("Ventricolo sinistro:")
    lvCombo = qt.QComboBox()
    lvCombo.addItems(segmentNames)
    # Trova il miglior match per LV
    best_lv_match = self._findBestMatch(segmentNames, ["left", "lv", "ventricle"])
    if best_lv_match >= 0:
        lvCombo.setCurrentIndex(best_lv_match)
    
    # Combo box per il miocardio
    myoLabel = qt.QLabel("Miocardio:")
    myoCombo = qt.QComboBox()
    myoCombo.addItems(segmentNames)
    # Trova il miglior match per miocardio
    best_myo_match = self._findBestMatch(segmentNames, ["myo", "myocardium", "heart muscle"])
    if best_myo_match >= 0:
        myoCombo.setCurrentIndex(best_myo_match)
    
    # Aggiungi i widget al layout
    formLayout = qt.QFormLayout()
    formLayout.addRow(rvLabel, rvCombo)
    formLayout.addRow(lvLabel, lvCombo)
    formLayout.addRow(myoLabel, myoCombo)
    layout.addLayout(formLayout)
    
    # Pulsanti di conferma
    buttonBox = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
    buttonBox.accepted.connect(dialog.accept)
    buttonBox.rejected.connect(dialog.reject)
    layout.addWidget(buttonBox)
    
    # Mostra il dialogo
    result = dialog.exec_()
    
    if result == qt.QDialog.Accepted:
        # Aggiorna i campi di testo con i nomi selezionati
        self.rightVentricleNameEdit.text = rvCombo.currentText
        self.leftVentricleNameEdit.text = lvCombo.currentText
        self.myocardiumNameEdit.text = myoCombo.currentText
        
        qt.QMessageBox.information(None, "Completato", "Mappatura dei segmenti aggiornata. Usa questi nomi per il calcolo dei volumi.")

  def _findBestMatch(self, nameList, keywords):
    """Trova il miglior match in una lista di nomi basandosi su parole chiave"""
    bestScore = -1
    bestIndex = -1
    
    for i, name in enumerate(nameList):
        name = name.lower()
        score = 0
        for keyword in keywords:
            if keyword.lower() in name:
                score += 1
        
        if score > bestScore:
            bestScore = score
            bestIndex = i
    
    return bestIndex
  
  def onPhaseDetectionMethodChanged(self):
    """Gestisce il cambio del metodo di rilevamento delle fasi"""
    method = self.phaseDetectionMethodSelector.currentData
    
    # Aggiorna la UI in base al metodo selezionato
    if method == "manual":
      # Per la selezione manuale, abilita i selettori
      self.edvPhaseSelector.setEnabled(True)
      self.esvPhaseSelector.setEnabled(True)
    else:
      # Per i metodi automatici, disabilita i selettori (verranno impostati automaticamente)
      self.edvPhaseSelector.setEnabled(False)
      self.esvPhaseSelector.setEnabled(False)
    
    # Se abbiamo già calcolato i volumi, aggiorna i risultati con il nuovo metodo
    if hasattr(self, 'volumeData') and self.volumeData and 'lv_volume' in self.volumeData:
      self.detectCardiacPhases()
      self.onUpdateResults()
  
  def onDebugButton(self):
    """Funzione di debug per controllare la segmentazione"""
    segNode = self.segmentationSelector.currentNode()
    if not segNode:
      qt.QMessageBox.warning(None, "Avviso", "Nessuna segmentazione selezionata.")
      return
    
    # Verifica se abbiamo una sequenza di volume selezionata
    volumeSeqNode = None
    if self.useVolumeSequenceCheckBox.checked:
      volumeSeqNode = self.volumeSequenceSelector.currentNode()
    
    # Ottieni i nomi dei segmenti personalizzati
    rightVentricleName = self.rightVentricleNameEdit.text
    leftVentricleName = self.leftVentricleNameEdit.text
    myocardiumName = self.myocardiumNameEdit.text
    
    # Visualizza un dialogo con le informazioni di debug
    infoText = self.logic.getSegmentationInfo(segNode, 
                                            rightVentricleName=rightVentricleName,
                                            leftVentricleName=leftVentricleName,
                                            myocardiumName=myocardiumName)
    
    # Aggiungi informazioni sulla sequenza se disponibile
    if volumeSeqNode:
      infoText += "\n\n--- INFORMAZIONI SEQUENZA ---\n"
      infoText += f"Nome sequenza: {volumeSeqNode.GetName()}\n"
      infoText += f"ID sequenza: {volumeSeqNode.GetID()}\n"
      infoText += f"Numero di elementi nella sequenza: {volumeSeqNode.GetNumberOfDataNodes()}\n"
      
      # Get the sequence browser module logic
      seqBrowserLogic = slicer.modules.sequences.logic()
      
      # Find browser nodes
      browserNode = seqBrowserLogic.GetFirstBrowserNodeForSequenceNode(volumeSeqNode)
      segBrowserNode = seqBrowserLogic.GetFirstBrowserNodeForProxyNode(segNode)
      
      if browserNode:
        infoText += f"\nBrowser per sequenza volume: {browserNode.GetName()}\n"
        infoText += f"Numero di nodi proxy: {browserNode.GetNumberOfProxyNodes()}\n"
      
      if segBrowserNode:
        if segBrowserNode == browserNode:
          infoText += f"\nLa segmentazione è sincronizzata con la sequenza volume (stesso browser)\n"
        else:
          infoText += f"\nLa segmentazione è in un browser diverso: {segBrowserNode.GetName()}\n"
      else:
        infoText += f"\nLa segmentazione NON è sincronizzata con alcuna sequenza\n"
    
    # Crea un dialogo per visualizzare le informazioni
    infoDialog = qt.QDialog()
    infoDialog.setWindowTitle("Informazioni Segmentazione")
    infoDialog.setMinimumWidth(600)
    infoDialog.setMinimumHeight(400)
    
    layout = qt.QVBoxLayout(infoDialog)
    
    # Aggiungi un widget di testo per visualizzare le informazioni
    textEdit = qt.QTextEdit()
    textEdit.setPlainText(infoText)
    textEdit.setReadOnly(True)
    layout.addWidget(textEdit)
    
    # Aggiungi pulsante OK
    okButton = qt.QPushButton("OK")
    okButton.clicked.connect(infoDialog.accept)  # No arguments
    layout.addWidget(okButton)
    
    infoDialog.exec_()
  
  def detectCardiacPhases(self):
    """Rileva le fasi di telediastole e telesistole in base al metodo selezionato"""
    method = self.phaseDetectionMethodSelector.currentData
    
    if method == "auto":
      # Metodo semplice basato sui picchi di volume
      edv_phase = np.argmax(self.volumeData['lv_volume'])
      esv_phase = np.argmin(self.volumeData['lv_volume'])
    
    elif method == "advanced":
      # Metodo avanzato che considera più fattori
      edv_phase, esv_phase = self.logic.detect_cardiac_phases_robust(
          self.volumeData['lv_volume'], 
          self.volumeData['rv_volume'],
          self.volumeData['myocardial_mass']
      )
    
    else:  # manual
      # Usa le fasi selezionate manualmente dall'utente
      return
    
    # Imposta i valori nei selettori
    self.edvPhaseSelector.setCurrentIndex(edv_phase)
    self.esvPhaseSelector.setCurrentIndex(esv_phase)
    
    # Aggiorna i dati
    self.volumeData['selected_edv_phase'] = edv_phase
    self.volumeData['selected_esv_phase'] = esv_phase
  
  def onCalculateButton(self):
    """Calcola volumi e metriche per ogni fase temporale"""
    # Ottieni il nodo di segmentazione selezionato
    segNode = self.segmentationSelector.currentNode()
    
    # Controlla se stiamo usando una sequenza di volume
    useVolumeSequence = self.useVolumeSequenceCheckBox.checked
    volumeSequenceNode = self.volumeSequenceSelector.currentNode() if useVolumeSequence else None
    
    # Se stiamo usando una sequenza di volume, verificala
    if useVolumeSequence:
        if not volumeSequenceNode:
            qt.QMessageBox.warning(None, "Avviso", "Seleziona una sequenza di volume.")
            return
        
        # Verifica se abbiamo scelto di creare una segmentazione sincronizzata
        createSyncSegmentation = self.createSyncSegmentationCheckBox.checked
        
        if createSyncSegmentation or not segNode:
            # Crea una nuova segmentazione sincronizzata
            segNode = self._createSynchronizedSegmentation(volumeSequenceNode)
            if not segNode:
                qt.QMessageBox.warning(None, "Errore", "Impossibile creare una segmentazione sincronizzata.")
                return
            # Aggiorna il selettore
            self.segmentationSelector.setCurrentNode(segNode)
            
            # Informa l'utente
            qt.QMessageBox.information(None, "Informazione", 
                          "È stata creata una nuova segmentazione sincronizzata con la sequenza.\n"
                          "Prima di continuare, usa l'Editor di Segmentazione per creare i tuoi segmenti.\n\n"
                          "Quando hai finito, premi di nuovo 'Calcola metriche cardiache'.")
            return
        else:
            # Valida la compatibilità tra segmentazione e sequenza
            if not self.validateSequenceAndSegmentation():
                return
    
    # Ottieni i nomi personalizzati dei segmenti
    rightVentricleName = self.rightVentricleNameEdit.text
    leftVentricleName = self.leftVentricleNameEdit.text
    myocardiumName = self.myocardiumNameEdit.text
    
    # Verifica che la segmentazione contenga i segmenti necessari
    if not self.logic.checkRequiredSegments(segNode, 
                                          rightVentricleName=rightVentricleName,
                                          leftVentricleName=leftVentricleName, 
                                          myocardiumName=myocardiumName):
        qt.QMessageBox.warning(None, "Avviso", 
                           "La segmentazione non contiene tutti i segmenti necessari.\n\n"
                           "Usa il pulsante 'Rileva nomi dalla segmentazione corrente' per mappare i segmenti esistenti.")
        return
    
    # Ottieni il numero di fasi
    numPhases = self.numPhasesSpinBox.value
    
    # Mostra una barra di progresso
    progress = qt.QProgressDialog("Calcolo metriche...", "Annulla", 0, numPhases, slicer.util.mainWindow())
    progress.setWindowModality(qt.Qt.WindowModal)
    progress.show()
    
    # Prepara la tabella
    self.resultsTable.setRowCount(numPhases)
    self.volumeData = {
      'phase': list(range(numPhases)),
      'rv_volume': [],
      'lv_volume': [],
      'myocardial_mass': [],
      'lv_stroke_volume': [0] * numPhases,
      'rv_stroke_volume': [0] * numPhases,
      'selected_edv_phase': 0,
      'selected_esv_phase': 0
    }
    
    # Svuota i selettori di fase
    self.edvPhaseSelector.clear()
    self.esvPhaseSelector.clear()
    
    # Inizializza variabili per la gestione della sequenza
    sequenceBrowser = None
    originalIndex = 0
    
    # Get the sequence browser module logic
    seqBrowserLogic = slicer.modules.sequences.logic()
    
    # Configura il browser di sequenza se necessario
    if useVolumeSequence and volumeSequenceNode:
      # Trova o crea un browser per la sequenza
      sequenceBrowser = self._findOrCreateBrowserForSequence(volumeSequenceNode)
      if sequenceBrowser:
        # Salva l'indice originale
        originalIndex = sequenceBrowser.GetSelectedItemNumber()
        # Assicurati che il browser sia attivo ma non in riproduzione
        sequenceBrowser.SetPlaybackActive(False)
    else:
      # Se non stiamo usando una sequenza di volume, controlliamo se la segmentazione
      # è collegata a un browser di sequenza (caso multivolume)
      sequenceBrowser = seqBrowserLogic.GetFirstBrowserNodeForProxyNode(segNode)
      if sequenceBrowser:
        originalIndex = sequenceBrowser.GetSelectedItemNumber()
    
    # Per ogni fase, calcola i volumi
    for phase in range(numPhases):
      progress.setValue(phase)
      
      # Imposta il frame corrente per questa fase
      if sequenceBrowser:
        sequenceBrowser.SetPlaybackActive(False)
        originalItemNumber = sequenceBrowser.GetSelectedItemNumber()
        sequenceBrowser.SetSelectedItemNumber(phase)
        currentItemNumber = sequenceBrowser.GetSelectedItemNumber()
        print(f"Richiesto frame {phase}, ottenuto frame {currentItemNumber}")
        
        # Aggiorna la visualizzazione e attendi
        slicer.app.processEvents()
        # Aggiungi un breve ritardo per assicurarti che la visualizzazione sia aggiornata
        time.sleep(0.1)
        slicer.app.processEvents()
      
      # Forza l'aggiornamento della segmentazione
      segNode.Modified()
      slicer.app.processEvents()
      
      # Calcola i volumi per questa fase
      rv_vol = self.logic.calculateSegmentVolume(segNode, rightVentricleName)
      lv_vol = self.logic.calculateSegmentVolume(segNode, leftVentricleName)
      myocardial_vol = self.logic.calculateSegmentVolume(segNode, myocardiumName)
      
      # Converti volume miocardico in massa (assumendo densità = 1.05 g/ml)
      myocardial_mass = myocardial_vol * 1.05
      
      print(f"Fase {phase}: RV={rv_vol:.2f} ml, LV={lv_vol:.2f} ml, Myo={myocardial_mass:.2f} g")
      
      # Salva i dati
      self.volumeData['rv_volume'].append(rv_vol)
      self.volumeData['lv_volume'].append(lv_vol)
      self.volumeData['myocardial_mass'].append(myocardial_mass)
      
      # Popola la tabella
      self.resultsTable.setItem(phase, 0, qt.QTableWidgetItem(str(phase)))
      self.resultsTable.setItem(phase, 1, qt.QTableWidgetItem(f"{rv_vol:.2f}"))
      self.resultsTable.setItem(phase, 2, qt.QTableWidgetItem(f"{lv_vol:.2f}"))
      self.resultsTable.setItem(phase, 3, qt.QTableWidgetItem(f"{myocardial_mass:.2f}"))
      self.resultsTable.setItem(phase, 4, qt.QTableWidgetItem("0.00"))  # Inizializza a zero
      
      # Aggiungi questa fase ai selettori
      self.edvPhaseSelector.addItem(f"Fase {phase}")
      self.esvPhaseSelector.addItem(f"Fase {phase}")
      
      if progress.wasCanceled:  # Senza parentesi
        break
    
    # Ripristina il frame originale
    if sequenceBrowser:
      sequenceBrowser.SetSelectedItemNumber(originalIndex)
    
    progress.setValue(numPhases)
    
    # Rileva le fasi cardiache in base al metodo selezionato
    self.detectCardiacPhases()
    
    self.updateResultsButton.enabled = True
    self.exportButton.enabled = True
    
    # Aggiorniamo subito i risultati con le fasi selezionate
    self.onUpdateResults()
    
  def onUpdateResults(self):
    """Aggiorna i risultati in base alle fasi di telediastole e telesistole selezionate"""
    if not self.volumeData['phase'] or self.edvPhaseSelector.count == 0:
      return
    
    # Ottieni le fasi selezionate
    edv_index = self.edvPhaseSelector.currentIndex
    esv_index = self.esvPhaseSelector.currentIndex
    
    # Aggiorna le fasi selezionate nei dati
    self.volumeData['selected_edv_phase'] = edv_index
    self.volumeData['selected_esv_phase'] = esv_index
    
    # Calcola stroke volume per ventricolo sinistro
    if self.volumeData['lv_volume']:
      lv_edv = self.volumeData['lv_volume'][edv_index]
      lv_esv = self.volumeData['lv_volume'][esv_index]
      lv_sv = lv_edv - lv_esv
      
      # Calcola anche lo stroke volume per il ventricolo destro
      rv_edv = self.volumeData['rv_volume'][edv_index]
      rv_esv = self.volumeData['rv_volume'][esv_index]
      rv_sv = rv_edv - rv_esv
      
      # Aggiorna tutti i valori di stroke volume
      for i in range(len(self.volumeData['phase'])):
        # Memorizza i valori di SV per LV e RV
        self.volumeData['lv_stroke_volume'][i] = lv_sv if i == edv_index else 0
        self.volumeData['rv_stroke_volume'][i] = rv_sv if i == edv_index else 0
        
        # Aggiorna la cella nella tabella
        self.resultsTable.setItem(i, 4, qt.QTableWidgetItem(f"{lv_sv:.2f}"))
    
  def onExportButton(self):
    """Esporta i risultati in CSV e PDF"""
    if not self.volumeData['phase']:
        qt.QMessageBox.warning(None, "Avviso", "Nessun dato da esportare")
        return
        
    # Chiedi il percorso del file
    fileName = qt.QFileDialog.getSaveFileName(qt.QFileDialog(), "Salva analisi", 
                                             os.path.join(self.outputDirSelector.currentPath, "cardiac_analysis"),
                                             "Tutti i file (*.*)")
    if not fileName:
        return
    
    # Assicurati che abbiamo l'estensione corretta per CSV e PDF
    csvFileName = fileName + ".csv" if not fileName.endswith(".csv") else fileName
    pdfFileName = fileName.replace(".csv", "") + ".pdf"
    
    # Ottieni il nome del paziente
    patientName = self.patientNameEdit.text
    if not patientName:
        patientName = "Anonimo"
    
    # Crea DataFrame pandas
    df = pd.DataFrame({
        'Fase': self.volumeData['phase'],
        'Volume ventricolo destro (ml)': self.volumeData['rv_volume'],
        'Volume ventricolo sinistro (ml)': self.volumeData['lv_volume'],
        'Massa miocardica (g)': self.volumeData['myocardial_mass'],
        'Stroke volume VD (ml)': self.volumeData['rv_stroke_volume'],
        'Stroke volume VS (ml)': self.volumeData['lv_stroke_volume']
    })
    
    # Ottieni le fasi selezionate
    edv_phase = self.volumeData['selected_edv_phase']
    esv_phase = self.volumeData['selected_esv_phase']
    
    # Aggiungi righe di riepilogo in fondo al file CSV
    summary_data = pd.DataFrame([
        {"Descrizione": "EDV (Ventricolo destro)", "Valore": self.volumeData['rv_volume'][edv_phase], "Unità": "ml"},
        {"Descrizione": "ESV (Ventricolo destro)", "Valore": self.volumeData['rv_volume'][esv_phase], "Unità": "ml"},
        {"Descrizione": "Stroke Volume (Ventricolo destro)", 
         "Valore": self.volumeData['rv_volume'][edv_phase] - self.volumeData['rv_volume'][esv_phase], "Unità": "ml"},
        {"Descrizione": "Frazione di eiezione (Ventricolo destro)", 
         "Valore": ((self.volumeData['rv_volume'][edv_phase] - self.volumeData['rv_volume'][esv_phase]) / 
                    self.volumeData['rv_volume'][edv_phase]) * 100 if self.volumeData['rv_volume'][edv_phase] > 0 else 0, 
         "Unità": "%"},
        {"Descrizione": "EDV (Ventricolo sinistro)", "Valore": self.volumeData['lv_volume'][edv_phase], "Unità": "ml"},
        {"Descrizione": "ESV (Ventricolo sinistro)", "Valore": self.volumeData['lv_volume'][esv_phase], "Unità": "ml"},
        {"Descrizione": "Stroke Volume (Ventricolo sinistro)", 
         "Valore": self.volumeData['lv_volume'][edv_phase] - self.volumeData['lv_volume'][esv_phase], "Unità": "ml"},
        {"Descrizione": "Frazione di eiezione (Ventricolo sinistro)", 
         "Valore": ((self.volumeData['lv_volume'][edv_phase] - self.volumeData['lv_volume'][esv_phase]) / 
                    self.volumeData['lv_volume'][edv_phase]) * 100 if self.volumeData['lv_volume'][edv_phase] > 0 else 0, 
         "Unità": "%"},
        {"Descrizione": "Massa miocardica media", 
         "Valore": np.mean(self.volumeData['myocardial_mass']), "Unità": "g"}
    ])
    
    # Salva il CSV
    csvSaved = False
    try:
        # Salva i dati principali
        df.to_csv(csvFileName, index=False)
        
        # Aggiungi una riga vuota
        with open(csvFileName, 'a') as f:
            f.write("\n\nRiepilogo metriche cardiache:\n")
        
        # Salva il riepilogo
        summary_data.to_csv(csvFileName, index=False, mode='a')
        csvSaved = True
    except Exception as e:
        print(f"Errore nel salvataggio CSV: {str(e)}")
    
    # Genera anche il PDF con grafici
    pdfSaved = False
    try:
        # Verifica che FPDF sia disponibile
        try:
            from fpdf import FPDF
        except ImportError:
            import pip
            pip.main(['install', 'fpdf'])
            from fpdf import FPDF
        
        # 1. Crea un grafico come file immagine temporaneo
        import matplotlib
        matplotlib.use('Agg')  # Usa il backend non interattivo
        
        # Crea una cartella temporanea se non esiste
        temp_dir = os.path.join(os.path.dirname(pdfFileName), "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Genera il grafico dei volumi come immagine
        chart_file = os.path.join(temp_dir, "cardiac_volumes.png")
        plt.figure(figsize=(8, 6))
        plt.plot(self.volumeData['phase'], self.volumeData['rv_volume'], 'b-', label='Ventricolo destro')
        plt.plot(self.volumeData['phase'], self.volumeData['lv_volume'], 'r-', label='Ventricolo sinistro')
        
        # Evidenzia EDV e ESV
        plt.plot(edv_phase, self.volumeData['lv_volume'][edv_phase], 'ro', markersize=8, label='EDV')
        plt.plot(esv_phase, self.volumeData['lv_volume'][esv_phase], 'go', markersize=8, label='ESV')
        
        plt.title('Volumi ventricolari')
        plt.xlabel('Fase temporale')
        plt.ylabel('Volume (ml)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Salva l'immagine
        plt.savefig(chart_file)
        plt.close()
        
        # 2. Crea il PDF con FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # Intestazione
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Dati volumetrici cardiaci", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Data analisi: {time.strftime('%d/%m/%Y')}", 0, 1)
        pdf.cell(0, 10, f"Paziente: {patientName}", 0, 1)
        pdf.ln(5)
        
        # Sommario delle metriche
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Sommario dell'analisi", 0, 1)
        pdf.ln(5)
        
        # Ventricolo destro
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Ventricolo destro:", 0, 1)
        pdf.set_font("Arial", "", 12)
        
        rv_edv = self.volumeData['rv_volume'][edv_phase]
        rv_esv = self.volumeData['rv_volume'][esv_phase]
        rv_sv = rv_edv - rv_esv
        rv_ef = (rv_sv / rv_edv) * 100 if rv_edv > 0 else 0
        
        pdf.cell(0, 8, f"- EDV (Fase {edv_phase}): {rv_edv:.2f} ml", 0, 1)
        pdf.cell(0, 8, f"- ESV (Fase {esv_phase}): {rv_esv:.2f} ml", 0, 1)
        pdf.cell(0, 8, f"- Stroke Volume stimato: {rv_sv:.2f} ml", 0, 1)
        pdf.cell(0, 8, f"- Frazione di eiezione stimata: {rv_ef:.1f}%", 0, 1)
        pdf.ln(5)
        
        # Ventricolo sinistro
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Ventricolo sinistro:", 0, 1)
        pdf.set_font("Arial", "", 12)
        
        lv_edv = self.volumeData['lv_volume'][edv_phase]
        lv_esv = self.volumeData['lv_volume'][esv_phase]
        lv_sv = lv_edv - lv_esv
        lv_ef = (lv_sv / lv_edv) * 100 if lv_edv > 0 else 0
        
        pdf.cell(0, 8, f"- EDV (Fase {edv_phase}): {lv_edv:.2f} ml", 0, 1)
        pdf.cell(0, 8, f"- ESV (Fase {esv_phase}): {lv_esv:.2f} ml", 0, 1)
        pdf.cell(0, 8, f"- Stroke Volume stimato: {lv_sv:.2f} ml", 0, 1)
        pdf.cell(0, 8, f"- Frazione di eiezione stimata: {lv_ef:.1f}%", 0, 1)
        pdf.ln(5)
        
        # Massa miocardica
        pdf.cell(0, 10, f"Massa miocardica media: {np.mean(self.volumeData['myocardial_mass']):.2f} g", 0, 1)
        pdf.ln(5)
        
        # Aggiungi il grafico
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Grafico dei volumi ventricolari", 0, 1)
        pdf.image(chart_file, x = 10, y = None, w = 180)
        pdf.ln(5)
        
        # Aggiungi una nuova pagina per la tabella
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Tabella dei valori per fase", 0, 1)
        pdf.ln(5)
        
        # Crea l'intestazione della tabella
        pdf.set_font("Arial", "B", 10)
        col_width = 35
        pdf.cell(20, 10, "Fase", 1, 0, "C")
        pdf.cell(col_width, 10, "Vol. Ventr. Dx (ml)", 1, 0, "C")
        pdf.cell(col_width, 10, "Vol. Ventr. Sin (ml)", 1, 0, "C")
        pdf.cell(col_width, 10, "Massa Miocardica (g)", 1, 0, "C")
        pdf.cell(col_width, 10, "Stroke Volume (ml)", 1, 1, "C")
        
        # Aggiungi i dati della tabella
        pdf.set_font("Arial", "", 10)
        for i in range(len(self.volumeData['phase'])):
            pdf.cell(20, 10, str(self.volumeData['phase'][i]), 1, 0, "C")
            pdf.cell(col_width, 10, f"{self.volumeData['rv_volume'][i]:.2f}", 1, 0, "C")
            pdf.cell(col_width, 10, f"{self.volumeData['lv_volume'][i]:.2f}", 1, 0, "C")
            pdf.cell(col_width, 10, f"{self.volumeData['myocardial_mass'][i]:.2f}", 1, 0, "C")
            pdf.cell(col_width, 10, f"{self.volumeData['stroke_volume'][i]:.2f}", 1, 1, "C")
        
        # Salva il PDF
        pdf.output(pdfFileName)
        
        # Pulizia: rimuovi il file immagine temporaneo
        try:
            os.remove(chart_file)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
        
        pdfSaved = True
    except Exception as e:
        print(f"Errore nella creazione del PDF: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Informa l'utente
    if csvSaved and pdfSaved:
        qt.QMessageBox.information(None, "Completato", 
                                 f"L'analisi è stata esportata con successo:\n- CSV: {csvFileName}\n- PDF: {pdfFileName}")
    elif csvSaved:
        qt.QMessageBox.warning(None, "Attenzione", 
                             f"I dati sono stati salvati nel file CSV: {csvFileName}\n"
                             f"Non è stato possibile creare il PDF.")
    elif pdfSaved:
        qt.QMessageBox.warning(None, "Attenzione", 
                             f"È stato creato il PDF: {pdfFileName}\n"
                             f"Non è stato possibile salvare il file CSV.")
    else:
        qt.QMessageBox.critical(None, "Errore", 
                              f"Si è verificato un errore durante il salvataggio dei dati.")

class CardiacVolumeAnalysisLogic(ScriptedLoadableModuleLogic):
  """Implementa la logica del modulo"""
  
  def getSegmentationInfo(self, segmentationNode, rightVentricleName="right ventricle of heart", 
                         leftVentricleName="left ventricle of heart", myocardiumName="myocardium"):
    """Ottiene informazioni dettagliate sulla segmentazione per debug"""
    if not segmentationNode:
        return "Nessuna segmentazione selezionata."
    
    infoText = f"Nome segmentazione: {segmentationNode.GetName()}\n"
    infoText += f"ID segmentazione: {segmentationNode.GetID()}\n\n"
    
    segmentation = segmentationNode.GetSegmentation()
    infoText += f"Numero di segmenti: {segmentation.GetNumberOfSegments()}\n\n"
    
    # Ottieni informazioni su ciascun segmento
    infoText += "Dettagli segmenti:\n"
    segmentIDs = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(segmentIDs)
    
    for i in range(segmentIDs.GetNumberOfValues()):
        segmentID = segmentIDs.GetValue(i)
        segment = segmentation.GetSegment(segmentID)
        
        infoText += f"\nSegmento #{i+1}:\n"
        infoText += f"  ID: {segmentID}\n"
        infoText += f"  Nome: {segment.GetName()}\n"
        
        # Evita di chiamare metodi che non esistono
        infoText += f"  Rappresentazioni disponibili:\n"
    
    # Controlla i nomi attesi
    infoText += "\nVerifica nomi segmenti richiesti:\n"
    requiredSegments = [rightVentricleName, leftVentricleName, myocardiumName]
    for segName in requiredSegments:
        segId = segmentation.GetSegmentIdBySegmentName(segName)
        infoText += f"  {segName}: {'Trovato (ID: ' + segId + ')' if segId else 'NON TROVATO'}\n"
    
    return infoText
  
  def checkRequiredSegments(self, segmentationNode, rightVentricleName="right ventricle of heart", 
                           leftVentricleName="left ventricle of heart", myocardiumName="myocardium"):
    """Verifica se la segmentazione contiene i segmenti necessari"""
    if not segmentationNode:
      return False
    
    requiredSegments = [rightVentricleName, leftVentricleName, myocardiumName]
    segmentation = segmentationNode.GetSegmentation()
    
    for segmentName in requiredSegments:
      segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
      if not segmentId:
        print(f"Segmento mancante: {segmentName}")
        return False
    
    return True
  
  def detect_cardiac_phases_robust(self, lv_volumes, rv_volumes=None, myocardial_volumes=None):
    """
    Rileva le fasi di telediastole (EDV) e telesistole (ESV) in modo robusto
    usando multiple metriche se disponibili.
    """
    num_phases = len(lv_volumes)
    
    # Calcola i candidati basati sul volume ventricolare sinistro
    lv_edv_idx = np.argmax(lv_volumes)
    lv_esv_idx = np.argmin(lv_volumes)
    
    # Se disponibili, considera anche i volumi del ventricolo destro
    if rv_volumes is not None:
        rv_edv_idx = np.argmax(rv_volumes)
        rv_esv_idx = np.argmin(rv_volumes)
        
        # Calcola la somma dei volumi ventricolari
        total_volumes = [lv + rv for lv, rv in zip(lv_volumes, rv_volumes)]
        total_edv_idx = np.argmax(total_volumes)
        total_esv_idx = np.argmin(total_volumes)
    else:
        rv_edv_idx = lv_edv_idx
        rv_esv_idx = lv_esv_idx
        total_edv_idx = lv_edv_idx
        total_esv_idx = lv_esv_idx
    
    # Calcola i punteggi per ogni fase come potenziale EDV
    edv_scores = np.zeros(num_phases)
    esv_scores = np.zeros(num_phases)
    
    for i in range(num_phases):
        # Calcola il punteggio EDV
        edv_scores[i] += (lv_volumes[i] / max(lv_volumes)) * 3  # Peso maggiore per LV
        
        if rv_volumes is not None:
            edv_scores[i] += (rv_volumes[i] / max(rv_volumes)) * 2
        
        # Aggiungi un bonus se questa fase è vicina ai picchi identificati
        if abs(i - lv_edv_idx) <= 1:
            edv_scores[i] += 1
        if abs(i - rv_edv_idx) <= 1:
            edv_scores[i] += 1
        if abs(i - total_edv_idx) <= 1:
            edv_scores[i] += 1
            
        # Calcola il punteggio ESV
        normalized_lv = 1 - (lv_volumes[i] / max(lv_volumes))
        esv_scores[i] += normalized_lv * 3  # Peso maggiore per LV
        
        if rv_volumes is not None:
            normalized_rv = 1 - (rv_volumes[i] / max(rv_volumes))
            esv_scores[i] += normalized_rv * 2
        
        # Aggiungi un bonus se questa fase è vicina ai minimi identificati
        if abs(i - lv_esv_idx) <= 1:
            esv_scores[i] += 1
        if abs(i - rv_esv_idx) <= 1:
            esv_scores[i] += 1
        if abs(i - total_esv_idx) <= 1:
            esv_scores[i] += 1
    
    # Trova le fasi con i punteggi più alti
    edv_phase = np.argmax(edv_scores)
    esv_phase = np.argmax(esv_scores)
    
    # Controlla se le fasi sono vicine l'una all'altra (potrebbe indicare un errore)
    if abs(edv_phase - esv_phase) < num_phases / 4:
        print("Attenzione: le fasi EDV ed ESV identificate sono insolitamente vicine")
    
    return edv_phase, esv_phase
  
  def calculateSegmentVolume(self, segmentationNode, segmentName):
    """Calcola il volume di un segmento usando Segment Statistics"""
    if not segmentationNode:
        return 0
    
    # Forza un aggiornamento dei nodi
    segmentationNode.Modified()
    slicer.app.processEvents()
    
    # Ottieni il segmento
    segmentation = segmentationNode.GetSegmentation()
    segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
    if not segmentId:
        print(f"Errore: segmento '{segmentName}' non trovato")
        return 0
    
    try:
        # Importa il modulo SegmentStatistics
        import SegmentStatistics
        
        # Crea un'istanza di SegmentStatisticsLogic
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        
        # Configura il calcolo
        paramNode = segStatLogic.getParameterNode()
        paramNode.SetParameter("Segmentation", segmentationNode.GetID())
        paramNode.SetParameter("LabelmapSegmentStatisticsPlugin.enabled", "True")
        paramNode.SetParameter("LabelmapSegmentStatisticsPlugin.volume_mm3.enabled", "True")
        
        # Calcola le statistiche
        segStatLogic.computeStatistics()
        
        # Ottieni il volume direttamente dalle statistiche
        stats = segStatLogic.getStatistics()
        for segId in stats["SegmentIDs"]:
            if segId == segmentId:
                volumeMm3 = stats[segId, "LabelmapSegmentStatisticsPlugin.volume_mm3"]
                volumeMl = volumeMm3 / 1000.0
                print(f"Volume calcolato per {segmentName}: {volumeMl} ml")
                return volumeMl
        
        # Se non troviamo il segmento nelle statistiche, proviamo l'altro metodo
        print(f"Segmento {segmentName} non trovato nelle statistiche, uso metodo alternativo")
        
        # Metodo alternativo
        # Crea una tabella temporanea per ottenere i risultati
        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        tableNode.SetHideFromEditors(True)
        
        # Esporta le statistiche nella tabella
        segStatLogic.exportToTable(tableNode)
        
        # Cerca il volume del segmento nella tabella
        table = tableNode.GetTable()
        volumeMm3 = 0
        
        for i in range(table.GetNumberOfRows()):
            rowSegmentName = table.GetValue(i, 0).ToString()
            # Confronta con il nome del segmento
            if rowSegmentName == segmentName:
                for j in range(table.GetNumberOfColumns()):
                    colName = table.GetColumnName(j)
                    if colName == "Volume mm3":
                        volumeMm3 = float(table.GetValue(i, j).ToDouble())
                        break
                break
        
        # Pulisci
        slicer.mrmlScene.RemoveNode(tableNode)
        
        # Converti in ml
        volumeMl = volumeMm3 / 1000.0
        print(f"Volume calcolato con metodo alternativo per {segmentName}: {volumeMl} ml")
        return volumeMl
        
    except Exception as e:
        print(f"Errore nel calcolo del volume: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ultimo tentativo: calcola il volume direttamente dalla geometria del segmento
        try:
            # Ottieni la rappresentazione binaria
            binaryLabelmapRepresentation = segmentation.GetSegment(segmentId).GetRepresentation(
                vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
            
            if binaryLabelmapRepresentation:
                # Calcola il volume dalla rappresentazione binaria
                volumeCC = slicer.vtkSegmentationCore.vtkSegmentationCore.ComputeSegmentVolumeFromBinaryLabelmap(binaryLabelmapRepresentation)
                print(f"Volume calcolato con metodo di emergenza per {segmentName}: {volumeCC} ml")
                return volumeCC
        except:
            pass
            
        return 0
  
