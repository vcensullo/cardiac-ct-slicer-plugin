import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np

class CTOptimizer(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "CT 4D Optimizer Simple"
    self.parent.categories = ["Utilities"]
    self.parent.dependencies = []
    self.parent.contributors = ["CT Optimizer contributor"]
    self.parent.helpText = "Riduce il peso dei dataset 4D CT tramite ridimensionamento e riduzione della profondità di bit."
    self.parent.acknowledgementText = "Per TotalSegmentator."

class CTOptimizerWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    
    # Usa il layout esistente invece di crearne uno nuovo
    
    # Frame per input
    inputFrame = qt.QFrame()
    inputFrame.setFrameStyle(qt.QFrame.StyledPanel | qt.QFrame.Plain)
    inputLayout = qt.QFormLayout(inputFrame)
    
    # Input
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.inputSelector.setToolTip("Seleziona sequenza da ottimizzare")
    inputLayout.addRow("Input:", self.inputSelector)
    
    # Nome output
    self.nameEdit = qt.QLineEdit("Sequenza_Ottimizzata")
    inputLayout.addRow("Nome output:", self.nameEdit)
    
    self.layout.addWidget(inputFrame)
    
    # Frame per parametri
    paramFrame = qt.QFrame()
    paramFrame.setFrameStyle(qt.QFrame.StyledPanel | qt.QFrame.Plain)
    paramLayout = qt.QFormLayout(paramFrame)
    
    # Fattore scala
    self.scaleSlider = ctk.ctkSliderWidget()
    self.scaleSlider.minimum = 0.1
    self.scaleSlider.maximum = 1.0
    self.scaleSlider.singleStep = 0.1
    self.scaleSlider.value = 0.5
    self.scaleSlider.setToolTip("Fattore di ridimensionamento (1.0 = dimensione originale)")
    paramLayout.addRow("Fattore scala:", self.scaleSlider)
    
    # Bit depth
    self.bitDepthCombo = qt.QComboBox()
    self.bitDepthCombo.addItem("8-bit", 8)
    self.bitDepthCombo.addItem("12-bit", 12)
    self.bitDepthCombo.addItem("16-bit (originale)", 16)
    self.bitDepthCombo.currentIndex = 1  # Default 12-bit
    paramLayout.addRow("Profondità bit:", self.bitDepthCombo)
    
    self.layout.addWidget(paramFrame)
    
    # Info stima dimensione
    self.sizeLabel = qt.QLabel("Dimensione stimata: --")
    self.layout.addWidget(self.sizeLabel)
    
    # Pulsante
    self.applyButton = qt.QPushButton("Ottimizza")
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)
    
    # Progress bar
    self.progressBar = qt.QProgressBar()
    self.progressBar.visible = False
    self.layout.addWidget(self.progressBar)
    
    # Connessioni
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.applyButton.connect("clicked(bool)", self.onApply)
    self.scaleSlider.connect("valueChanged(double)", self.updateSizeEstimate)
    self.bitDepthCombo.connect("currentIndexChanged(int)", self.updateSizeEstimate)
    
    # Aggiungi spazio vuoto alla fine
    self.layout.addStretch(1)
    
  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() is not None
    
    # Aggiorna nome automatico
    if self.inputSelector.currentNode():
      input_name = self.inputSelector.currentNode().GetName()
      if input_name:
        self.nameEdit.text = f"{input_name}_Ottimizzato"
      
    # Aggiorna stima dimensione
    self.updateSizeEstimate()
    
  def updateSizeEstimate(self):
    if not self.inputSelector.currentNode():
      self.sizeLabel.text = "Dimensione stimata: --"
      return
      
    try:
      inputNode = self.inputSelector.currentNode()
      
      # Conta frame e ottieni info volume
      num_frames = inputNode.GetNumberOfDataNodes()
      if num_frames == 0:
        self.sizeLabel.text = "Dimensione stimata: sequenza vuota"
        return
        
      # Ottieni dimensioni dal primo frame
      firstVolume = inputNode.GetNthDataNode(0)
      if not firstVolume or not firstVolume.IsA("vtkMRMLScalarVolumeNode"):
        self.sizeLabel.text = "Dimensione stimata: frame non valido"
        return
      
      # Calcola dimensioni e fattori
      dims = firstVolume.GetImageData().GetDimensions()
      scale_factor = self.scaleSlider.value
      bit_depth = self.bitDepthCombo.currentData
      original_bit_depth = 16  # Assumiamo 16-bit per CT
      
      # Calcola voxel
      original_voxels = dims[0] * dims[1] * dims[2] * num_frames
      new_voxels = original_voxels * (scale_factor ** 3)
      
      # Calcola MB
      original_mb = (original_voxels * original_bit_depth) / (8 * 1024 * 1024)
      new_mb = (new_voxels * bit_depth) / (8 * 1024 * 1024)
      
      # Aggiorna label
      reduction = (1 - (new_mb / original_mb)) * 100
      self.sizeLabel.text = f"Dimensione: {original_mb:.1f} MB → {new_mb:.1f} MB (-{reduction:.1f}%)"
      
    except Exception as e:
      self.sizeLabel.text = f"Errore stima: {str(e)}"
    
  def onApply(self):
    inputNode = self.inputSelector.currentNode()
    if not inputNode:
      return
      
    self.progressBar.visible = True
    self.progressBar.setValue(0)
    
    # Lista per tenere traccia dei nodi temporanei da pulire
    temp_nodes = []
    
    try:
      # Parametri
      scale_factor = self.scaleSlider.value
      bit_depth = self.bitDepthCombo.currentData
      output_name = self.nameEdit.text
      
      # Crea nodo output
      outputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
      outputNode.SetName(output_name)
      outputNode.SetIndexType(inputNode.GetIndexType())
      outputNode.SetIndexName(inputNode.GetIndexName())
      outputNode.SetIndexUnit(inputNode.GetIndexUnit())
      
      # Metadati
      outputNode.SetAttribute("CTOptimizer_ScaleFactor", str(scale_factor))
      outputNode.SetAttribute("CTOptimizer_BitDepth", str(bit_depth))
      
      # Conta frame
      num_frames = inputNode.GetNumberOfDataNodes()
      successful_frames = 0
      
      # Processa ogni frame
      for frame_idx in range(num_frames):
        # Aggiorna progresso
        progress = frame_idx / num_frames
        self.progressBar.setValue(int(progress * 100))
        slicer.app.processEvents()
        
        try:
          # Ottieni frame e indice
          input_volume = inputNode.GetNthDataNode(frame_idx)
          if not input_volume or not input_volume.IsA("vtkMRMLScalarVolumeNode"):
            continue
              
          index_value = inputNode.GetNthIndexValue(frame_idx)
          
          # Crea volume temporaneo
          temp_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
          temp_volume.SetName(f"Temp_{frame_idx}")
          temp_volume.Copy(input_volume)
          temp_nodes.append(temp_volume)
          
          # PASSO 1: Ridimensionamento se richiesto
          if scale_factor < 1.0:
            try:
              # Usa ResampleScalarVolume CLI
              resampled_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
              resampled_volume.SetName(f"Resampled_{frame_idx}")
              temp_nodes.append(resampled_volume)
              
              # Calcola nuovo spacing
              old_spacing = input_volume.GetSpacing()
              new_spacing = [s/scale_factor for s in old_spacing]
              
              # Parametri CLI
              params = {}
              params["inputVolume"] = temp_volume.GetID()
              params["outputVolume"] = resampled_volume.GetID()
              params["spacing"] = new_spacing
              params["interpolationType"] = "linear"
              
              # Esegui ricampionamento
              cliNode = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, params)
              temp_nodes.append(cliNode)
              
              # Verifica risultato e aggiorna
              if cliNode.GetStatus() == cliNode.Completed:
                temp_volume.Copy(resampled_volume)
            except Exception as e:
              print(f"Errore ridimensionamento frame {frame_idx}: {str(e)}")
          
          # PASSO 2: Riduzione bit depth
          if bit_depth < 16:
            try:
              # Ottieni array
              array = slicer.util.array(temp_volume.GetID())
              if array is not None:
                # Memorizza i valori originali min/max per metadati
                min_val = np.min(array)
                max_val = np.max(array)
                
                # Converti al nuovo bit depth
                if bit_depth == 8:
                  array_norm = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:  # 12 bit
                  array_norm = ((array - min_val) / (max_val - min_val) * 4095).astype(np.uint16)
                
                # Aggiorna array
                array[:] = array_norm
                temp_volume.Modified()
                
                # Salva metadati
                temp_volume.SetAttribute("CTOptimizer_OriginalMin", str(min_val))
                temp_volume.SetAttribute("CTOptimizer_OriginalMax", str(max_val))
                temp_volume.SetAttribute("CTOptimizer_BitDepth", str(bit_depth))
            except Exception as e:
              print(f"Errore riduzione bit frame {frame_idx}: {str(e)}")
          
          # PASSO 3: Aggiungi a sequenza
          new_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
          new_volume.Copy(temp_volume)
          temp_nodes.append(new_volume)
          
          outputNode.SetDataNodeAtValue(new_volume, index_value)
          
          # Incrementa contatore
          successful_frames += 1
          
        except Exception as e:
          print(f"Errore elaborazione frame {frame_idx}: {str(e)}")
      
      # Progresso completo
      self.progressBar.setValue(100)
      
      # Crea browser
      try:
        browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
        browser.SetName(output_name + "_Browser")
        slicer.modules.sequences.logic().AddSynchronizedNode(outputNode, None, browser)
        
        # Attiva browser
        slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
        slicer.modules.sequences.logic().UpdateAllProxyNodes()
        slicer.util.selectModule("Sequences")
      except Exception as e:
        print(f"Errore creazione browser: {str(e)}")
      
      # Messaggio completamento
      if successful_frames == num_frames:
        message = f"Ottimizzazione completata!\n\nSequenza: {output_name}\nFrame: {successful_frames}/{num_frames}"
      else:
        message = f"Ottimizzazione parziale!\n\nSequenza: {output_name}\nFrame elaborati: {successful_frames}/{num_frames}"
      
      slicer.util.infoDisplay(message)
      
    except Exception as e:
      slicer.util.errorDisplay(f"Errore generale: {str(e)}")
    finally:
      # Pulizia: rimuove tutti i nodi temporanei
      for node in temp_nodes:
        if node and slicer.mrmlScene.IsNodePresent(node):
          slicer.mrmlScene.RemoveNode(node)
      
      self.progressBar.visible = False

class CTOptimizerLogic(ScriptedLoadableModuleLogic):
  pass

class CTOptimizerTest(ScriptedLoadableModuleTest):
  def setUp(self):
    slicer.mrmlScene.Clear()
  
  def runTest(self):
    self.setUp()
    self.delayDisplay('Test completato!')