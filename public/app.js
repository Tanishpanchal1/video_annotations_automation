// AI Video Annotation Studio - Frontend Controller

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const imageSelect = document.getElementById('image-select');
  const audioSelect = document.getElementById('audio-select');
  const imageDrop = document.getElementById('image-drop');
  const audioDrop = document.getElementById('audio-drop');
  const imageFile = document.getElementById('image-file');
  const audioFile = document.getElementById('audio-file');
  const runBtn = document.getElementById('run-btn');
  
  const selectedImageDisplay = document.getElementById('selected-image-display');
  const selectedAudioDisplay = document.getElementById('selected-audio-display');
  const uploadedImageName = document.getElementById('uploaded-image-name');
  const uploadedAudioName = document.getElementById('uploaded-audio-name');
  
  const progressContainer = document.getElementById('progress-container');
  const videoContainer = document.getElementById('video-container');
  const videoPlaceholder = document.getElementById('video-placeholder');
  const videoPlayerWrapper = document.getElementById('video-player-wrapper');
  const outputVideo = document.getElementById('output-video');
  const downloadBtn = document.getElementById('download-btn');
  
  const consoleLogs = document.getElementById('console-logs');
  const clearTerminalBtn = document.getElementById('clear-terminal-btn');
  
  // Tab Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');
  const transcriptPlaceholder = document.getElementById('transcript-placeholder');
  const transcriptList = document.getElementById('transcript-list');
  const visionPlaceholder = document.getElementById('vision-placeholder');
  const visionInfo = document.getElementById('vision-info');
  const planPlaceholder = document.getElementById('plan-placeholder');
  const planList = document.getElementById('plan-list');
  
  // Inspector sub-elements
  const inspectorQuestion = document.getElementById('inspector-question');
  const inspectorSubject = document.getElementById('inspector-subject');
  const inspectorTopic = document.getElementById('inspector-topic');
  const inspectorCorrectAnswer = document.getElementById('inspector-correct-answer');
  const inspectorOptions = document.getElementById('inspector-options');
  const inspectorWorkingArea = document.getElementById('inspector-working-area');

  // Render progress elements
  const renderProgressWrapper = document.getElementById('render-progress-wrapper');
  const renderProgressFill = document.getElementById('render-progress-fill');
  const renderPercent = document.getElementById('render-percent');

  // App State
  let selectedImage = null;
  let selectedAudio = null;
  let eventSource = null;

  // Initialize: Load workspace assets
  loadAssets();

  // Tab switching logic
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));
      
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Fetch assets function
  async function loadAssets() {
    try {
      const res = await fetch('/api/assets');
      const data = await res.json();
      
      // Clear dropdowns except default
      imageSelect.innerHTML = '<option value="" disabled selected>Select from workspace...</option>';
      audioSelect.innerHTML = '<option value="" disabled selected>Select from workspace...</option>';
      
      data.images.forEach(img => {
        const option = document.createElement('option');
        option.value = img.path;
        option.textContent = `${img.name} (${img.type === 'workspace' ? 'Workspace' : 'Uploaded'})`;
        imageSelect.appendChild(option);
      });
      
      data.audios.forEach(aud => {
        const option = document.createElement('option');
        option.value = aud.path;
        option.textContent = `${aud.name} (${aud.type === 'workspace' ? 'Workspace' : 'Uploaded'})`;
        audioSelect.appendChild(option);
      });
    } catch (err) {
      logToConsole('System Error: Failed to load workspace assets: ' + err.message, 'error');
    }
  }

  // Handle dropdown selections
  imageSelect.addEventListener('change', (e) => {
    selectedImage = e.target.value;
    selectedImageDisplay.textContent = selectedImage;
    uploadedImageName.textContent = "No file chosen"; // Reset upload label
    checkValidity();
  });

  audioSelect.addEventListener('change', (e) => {
    selectedAudio = e.target.value;
    selectedAudioDisplay.textContent = selectedAudio;
    uploadedAudioName.textContent = "No file chosen"; // Reset upload label
    checkValidity();
  });

  // Drag & drop logic
  setupDragAndDrop(imageDrop, imageFile, 'image', uploadedImageName, (uploadedPath) => {
    selectedImage = uploadedPath;
    selectedImageDisplay.textContent = uploadedPath;
    imageSelect.value = ""; // Reset dropdown
    checkValidity();
  });

  setupDragAndDrop(audioDrop, audioFile, 'audio', uploadedAudioName, (uploadedPath) => {
    selectedAudio = uploadedPath;
    selectedAudioDisplay.textContent = uploadedPath;
    audioSelect.value = ""; // Reset dropdown
    checkValidity();
  });

  function setupDragAndDrop(dropZone, fileInput, fieldName, nameLabel, onUploadSuccess) {
    // Prevent defaults
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('highlight'), false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length) {
        fileInput.files = files;
        handleFileUpload(files[0], fieldName, nameLabel, onUploadSuccess);
      }
    });

    // Handle selected files
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        handleFileUpload(e.target.files[0], fieldName, nameLabel, onUploadSuccess);
      }
    });
  }

  async function handleFileUpload(file, fieldName, nameLabel, onUploadSuccess) {
    nameLabel.textContent = "Uploading: " + file.name + "...";
    logToConsole(`Uploading ${fieldName} file: ${file.name}...`);
    
    const formData = new FormData();
    formData.append(fieldName, file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) throw new Error("Upload response not OK");
      
      const data = await res.json();
      nameLabel.textContent = file.name + " (Uploaded)";
      logToConsole(`Upload complete! Saved as: ${data[fieldName]}`);
      onUploadSuccess(data[fieldName]);
      
      // Reload assets dropdown list to show new uploads
      loadAssets();
    } catch (err) {
      nameLabel.textContent = "Upload failed";
      logToConsole(`Upload failed for ${file.name}: ${err.message}`, 'error');
    }
  }

  function checkValidity() {
    if (selectedImage && selectedAudio) {
      runBtn.classList.remove('disabled');
      runBtn.removeAttribute('disabled');
    } else {
      runBtn.classList.add('disabled');
      runBtn.setAttribute('disabled', 'true');
    }
  }

  // Clear Terminal Button
  clearTerminalBtn.addEventListener('click', () => {
    consoleLogs.innerHTML = '';
  });

  // Run pipeline triggering
  runBtn.addEventListener('click', () => {
    if (!selectedImage || !selectedAudio) return;

    // Reset UI state
    progressContainer.classList.remove('hidden');
    videoPlaceholder.classList.remove('hidden');
    videoPlayerWrapper.classList.add('hidden');
    outputVideo.src = '';
    
    // Clear dynamic tabs
    transcriptList.innerHTML = '';
    transcriptList.classList.add('hidden');
    transcriptPlaceholder.classList.remove('hidden');
    
    visionInfo.classList.add('hidden');
    visionPlaceholder.classList.remove('hidden');
    inspectorOptions.innerHTML = '';
    
    planList.innerHTML = '';
    planList.classList.add('hidden');
    planPlaceholder.classList.remove('hidden');

    renderProgressWrapper.classList.add('hidden');
    renderProgressFill.style.width = '0%';
    renderPercent.textContent = '0%';

    // Reset steps nodes
    const steps = ['transcribe', 'analyze', 'plan', 'render', 'compose'];
    steps.forEach(step => {
      const el = document.getElementById(`step-${step}`);
      el.className = 'step-node'; // Remove running, done, error
      el.querySelector('.status-msg').textContent = 'Waiting...';
      const badges = document.getElementById(`${step}-badges`);
      if (badges) badges.innerHTML = '';
    });

    logToConsole('Starting annotation pipeline...', 'system');
    runBtn.classList.add('disabled');
    runBtn.setAttribute('disabled', 'true');

    // Establish SSE Connection
    const sseUrl = `/api/stream-progress?image=${encodeURIComponent(selectedImage)}&audio=${encodeURIComponent(selectedAudio)}`;
    eventSource = new EventSource(sseUrl);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'log') {
        // Log to panel
        let logClass = '';
        if (data.message.includes('error') || data.message.includes('RuntimeError') || data.message.includes('✗')) {
          logClass = 'error';
        } else if (data.message.startsWith('[stderr]')) {
          logClass = 'stderr';
        }
        logToConsole(data.message, logClass);
        
        // Parse logs to populate Details Inspector in real-time
        parseLogsForInspector(data.message);
      }
      
      else if (data.type === 'progress') {
        const stepNode = document.getElementById(`step-${data.step}`);
        if (!stepNode) return;

        if (data.status === 'running') {
          // Set current step to running, and mark previous steps as done if they were running
          stepNode.classList.add('running');
          stepNode.classList.remove('done');
          stepNode.querySelector('.status-msg').textContent = 'In progress...';
          
          // Auto-mark previous steps as done if we started a new one
          autoMarkPreviousDone(data.step);
        } 
        
        else if (data.status === 'done') {
          stepNode.classList.remove('running');
          stepNode.classList.add('done');
          stepNode.querySelector('.status-msg').textContent = 'Completed!';

          // Fill step details badges
          const badgeRow = document.getElementById(`${data.step}-badges`);
          if (badgeRow && data.detail) {
            badgeRow.innerHTML = '';
            Object.entries(data.detail).forEach(([key, val]) => {
              const badge = document.createElement('span');
              badge.className = 'step-badge';
              badge.textContent = `${key}: ${typeof val === 'number' ? val.toFixed(1) : val}`;
              badgeRow.appendChild(badge);
            });
          }
        }
      }
      
      else if (data.type === 'complete') {
        logToConsole(`Pipeline finished successfully! Output file is: ${data.output}`, 'system');
        
        // Mark remaining steps as done
        const steps = ['transcribe', 'analyze', 'plan', 'render', 'compose'];
        steps.forEach(step => {
          const el = document.getElementById(`step-${step}`);
          el.className = 'step-node done';
          el.querySelector('.status-msg').textContent = 'Completed!';
        });

        // Reveal video
        videoPlaceholder.classList.add('hidden');
        videoPlayerWrapper.classList.remove('hidden');
        outputVideo.src = `/workspace/${data.output}`;
        downloadBtn.href = `/workspace/${data.output}`;
        outputVideo.load();
        outputVideo.play().catch(e => console.log("Video auto-play blocked: ", e));

        eventSource.close();
        checkValidity(); // Re-enable generate button if inputs still valid
      }
      
      else if (data.type === 'error') {
        logToConsole(`Pipeline error: ${data.message}`, 'error');
        
        // Mark current/last active step as error
        const activeNode = document.querySelector('.step-node.running') || document.querySelector('.step-node:not(.done)');
        if (activeNode) {
          activeNode.classList.remove('running');
          activeNode.classList.add('error');
          activeNode.querySelector('.status-msg').textContent = 'Error occurred!';
        }

        eventSource.close();
        checkValidity();
      }
    };

    eventSource.onerror = (err) => {
      logToConsole('Connection error to progress server.', 'error');
      eventSource.close();
      checkValidity();
    };
  });

  // Auto-mark preceding pipeline steps as completed
  function autoMarkPreviousDone(currentStep) {
    const sequence = ['transcribe', 'analyze', 'plan', 'render', 'compose'];
    const idx = sequence.indexOf(currentStep);
    if (idx === -1) return;
    
    for (let i = 0; i < idx; i++) {
      const stepNode = document.getElementById(`step-${sequence[i]}`);
      if (stepNode && !stepNode.classList.contains('done')) {
        stepNode.classList.remove('running');
        stepNode.classList.add('done');
        stepNode.querySelector('.status-msg').textContent = 'Completed!';
      }
    }
  }

  // Print helper for console logger
  function logToConsole(message, className = '') {
    const line = document.createElement('div');
    line.className = 'log-line ' + className;
    line.textContent = message;
    consoleLogs.appendChild(line);
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
  }

  // Regex Parsers for Console Stream to feed details panels
  function parseLogsForInspector(line) {
    // 1. Transcription Segments: "[12.5s] This is a sentence"
    const transcriptRegex = /^\s*\[(\d+\.\d+)s\]\s*(.*)$/;
    const tMatch = line.match(transcriptRegex);
    if (tMatch) {
      transcriptPlaceholder.classList.add('hidden');
      transcriptList.classList.remove('hidden');
      
      const timeVal = parseFloat(tMatch[1]);
      const textVal = tMatch[2];
      
      const item = document.createElement('div');
      item.className = 'transcript-item';
      item.innerHTML = `
        <div class="transcript-time">${timeVal.toFixed(1)}s</div>
        <div class="transcript-text">${textVal}</div>
      `;
      transcriptList.appendChild(item);
    }

    // 2. Vision Data: "SUCCESS: Question: Find the distance between..."
    if (line.includes('SUCCESS: Question:')) {
      visionPlaceholder.classList.add('hidden');
      visionInfo.classList.remove('hidden');
      inspectorQuestion.textContent = line.substring(line.indexOf('Question:') + 9).trim();
    }
    
    // "SUCCESS: Subject: math | Topic: distance formula"
    const subjectTopicRegex = /SUCCESS:\s*Subject:\s*([^|]+)\|\s*Topic:\s*(.*)$/i;
    const stMatch = line.match(subjectTopicRegex);
    if (stMatch) {
      visionPlaceholder.classList.add('hidden');
      visionInfo.classList.remove('hidden');
      inspectorSubject.textContent = stMatch[1].trim();
      inspectorTopic.textContent = stMatch[2].trim();
    }
    
    // "SUCCESS: Correct answer: C"
    if (line.includes('SUCCESS: Correct answer:')) {
      visionPlaceholder.classList.add('hidden');
      visionInfo.classList.remove('hidden');
      const ans = line.substring(line.indexOf('Correct answer:') + 15).trim();
      inspectorCorrectAnswer.textContent = ans;

      // Populate mock option choices around it
      inspectorOptions.innerHTML = '';
      ['A', 'B', 'C', 'D'].forEach(opt => {
        const item = document.createElement('div');
        item.className = `option-item ${opt === ans ? 'correct' : ''}`;
        item.innerHTML = `
          <span class="option-lbl">Option ${opt}</span>
          <span class="option-txt">${opt === ans ? 'Correct answer choice detected by vision model' : 'Alternative option choice'}</span>
        `;
        inspectorOptions.appendChild(item);
      });
      inspectorWorkingArea.textContent = 'x: 30, y: 280, w: 600, h: 250 (standard working box)';
    }

    // 3. Planned Annotations: "t=13.4s -> [write_text] write formula"
    const planRegex = /^\s*t=(\d+\.?\d*)s\s*->\s*\[([^\]]+)\]\s*(.*)$/;
    const pMatch = line.match(planRegex);
    if (pMatch) {
      planPlaceholder.classList.add('hidden');
      planList.classList.remove('hidden');
      
      const timeVal = parseFloat(pMatch[1]);
      const typeVal = pMatch[2];
      const labelVal = pMatch[3];
      
      const item = document.createElement('div');
      item.className = 'plan-item';
      item.innerHTML = `
        <span class="plan-time">${timeVal.toFixed(1)}s</span>
        <span class="plan-type-badge">${typeVal}</span>
        <span class="plan-label">${labelVal}</span>
      `;
      planList.appendChild(item);
    }

    // 4. Rendering Progress bar: "10s / 71s rendered..."
    const renderRegex = /^\s*(\d+)s\s*\/\s*(\d+)s\s+rendered\.\.\./;
    const rMatch = line.match(renderRegex);
    if (rMatch) {
      const currentSec = parseInt(rMatch[1]);
      const totalSec = parseInt(rMatch[2]);
      if (totalSec > 0) {
        renderProgressWrapper.classList.remove('hidden');
        const percentage = Math.min(Math.round((currentSec / totalSec) * 100), 100);
        renderProgressFill.style.width = percentage + '%';
        renderPercent.textContent = percentage + '%';

        // Also update step status message
        const renderStep = document.getElementById('step-render');
        if (renderStep) {
          renderStep.querySelector('.status-msg').textContent = `Rendering: ${currentSec}s / ${totalSec}s (${percentage}%)`;
        }
      }
    }
  }
});
