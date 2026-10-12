/**
 * Bengal Scheme-to-Video AI Studio - Frontend Controller
 * Enforces strict input validation (no API key or document = no generation).
 * Live editable scenes, AI Critic audit inspector, Bengali TTS player, and exports.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const alertBox = document.getElementById('alertBox');
  const alertMessage = document.getElementById('alertMessage');

  const providerSelect = document.getElementById('providerSelect');
  const modelSelect = document.getElementById('modelSelect');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const toggleKeyBtn = document.getElementById('toggleKeyVisibility');
  const btnTestKey = document.getElementById('btnTestKey');
  const keyStatusText = document.getElementById('keyStatusText');

  // Populate models based on selected provider
  providerSelect.addEventListener('change', () => {
    modelSelect.innerHTML = '';
    if (providerSelect.value === 'gemini') {
      const geminiModels = [
        { val: 'gemini-3.5-flash', label: 'gemini-3.5-flash (Verified Active & Ultra-Fast)' },
        { val: 'gemini-3.5-flash-lite', label: 'gemini-3.5-flash-lite' },
        { val: 'gemini-flash-latest', label: 'gemini-flash-latest' },
        { val: 'gemini-3.6-flash', label: 'gemini-3.6-flash' },
        { val: 'gemini-3.7-flash', label: 'gemini-3.7-flash' },
        { val: 'gemini-3.8-flash', label: 'gemini-3.8-flash' }
      ];
      geminiModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.val;
        opt.textContent = m.label;
        modelSelect.appendChild(opt);
      });
    } else {
      const openAiModels = [
        { val: 'gpt-4o', label: 'gpt-4o (Omni flagship)' },
        { val: 'gpt-4o-mini', label: 'gpt-4o-mini (Fast & efficient)' }
      ];
      openAiModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.val;
        opt.textContent = m.label;
        modelSelect.appendChild(opt);
      });
    }
  });

  const tabDocFile = document.getElementById('tabDocFile');
  const tabDocText = document.getElementById('tabDocText');
  const fileUploadContainer = document.getElementById('fileUploadContainer');
  const textPasteContainer = document.getElementById('textPasteContainer');

  const docDropzone = document.getElementById('docDropzone');
  const schemeFileInput = document.getElementById('schemeFileInput');
  const docFileInfo = document.getElementById('docFileInfo');
  const docFileName = document.getElementById('docFileName');
  const schemeTextInput = document.getElementById('schemeTextInput');

  const photoDropzone = document.getElementById('photoDropzone');
  const groundingPhotosInput = document.getElementById('groundingPhotosInput');
  const photoPreviewGrid = document.getElementById('photoPreviewGrid');
  const notesInput = document.getElementById('notesInput');

  const btnGenerate = document.getElementById('btnGenerate');
  const loadingBox = document.getElementById('loadingBox');
  const loadingTitle = document.getElementById('loadingTitle');
  const loadingSub = document.getElementById('loadingSub');
  const stepDoc = document.getElementById('stepDoc');
  const stepPrompt = document.getElementById('stepPrompt');
  const stepCritic = document.getElementById('stepCritic');

  const resultsSection = document.getElementById('resultsSection');
  const displaySchemeTitle = document.getElementById('displaySchemeTitle');
  const displayDuration = document.getElementById('displayDuration');
  const displaySceneCount = document.getElementById('displaySceneCount');
  const displayGlobalGuidelines = document.getElementById('displayGlobalGuidelines');
  const displayGroundingRefs = document.getElementById('displayGroundingRefs');
  const scenesContainer = document.getElementById('scenesContainer');

  const criticAuditCard = document.getElementById('criticAuditCard');
  const criticScoreBadge = document.getElementById('criticScoreBadge');
  const criticNotesText = document.getElementById('criticNotesText');
  const criticIssuesContainer = document.getElementById('criticIssuesContainer');
  const btnRerunCritic = document.getElementById('btnRerunCritic');

  const btnExportJson = document.getElementById('btnExportJson');
  const btnCopyPrompts = document.getElementById('btnCopyPrompts');
  const btnSaveEdits = document.getElementById('btnSaveEdits');

  // Application State
  let currentPlan = null;
  let activeSchemeText = "";
  let uploadedPhotos = [];

  // Shared state for the Video Production Studio (video_studio.js)
  window.studioBridge = {
    getPlan: () => currentPlan,
    getPhotos: () => uploadedPhotos,
    getLlmKey: () => apiKeyInput.value.trim(),
    getLlmProvider: () => providerSelect.value,
    showAlert: (msg, ok) => showAlert(msg, ok),
  };

  // Show Alert
  function showAlert(msg, isSuccess = false) {
    alertBox.className = isSuccess ? 'alert-box success' : 'alert-box';
    alertMessage.textContent = msg;
    alertBox.classList.remove('hidden');
    alertBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideAlert() {
    alertBox.classList.add('hidden');
  }

  // Toggle API Key Visibility
  toggleKeyBtn.addEventListener('click', () => {
    if (apiKeyInput.type === 'password') {
      apiKeyInput.type = 'text';
      toggleKeyBtn.textContent = 'Hide';
    } else {
      apiKeyInput.type = 'password';
      toggleKeyBtn.textContent = 'Show';
    }
  });

  // Verify Key Button
  btnTestKey.addEventListener('click', async () => {
    hideAlert();
    const key = apiKeyInput.value.trim();
    if (!key) {
      showAlert("Please enter an API key first to test.");
      keyStatusText.textContent = "Key missing";
      keyStatusText.style.color = "#f43f5e";
      return;
    }

    keyStatusText.textContent = "Verifying...";
    keyStatusText.style.color = "#38bdf8";

    const formData = new FormData();
    formData.append("provider", providerSelect.value);
    formData.append("api_key", key);
    formData.append("model_name", modelSelect.value);

    try {
      const resp = await fetch("/api/validate-key", { method: "POST", body: formData });
      const data = await resp.json();
      if (resp.ok) {
        keyStatusText.textContent = "✓ Valid Key";
        keyStatusText.style.color = "#34d399";
        showAlert("API Key successfully verified!", true);
      } else {
        keyStatusText.textContent = "✗ Invalid Key";
        keyStatusText.style.color = "#f43f5e";
        showAlert(data.detail || "API Key verification failed.");
      }
    } catch (e) {
      keyStatusText.textContent = "✗ Verification Error";
      keyStatusText.style.color = "#f43f5e";
      showAlert("Network error during API key verification: " + e.message);
    }
  });

  // Tab switching for Document
  tabDocFile.addEventListener('click', () => {
    tabDocFile.classList.add('active');
    tabDocText.classList.remove('active');
    fileUploadContainer.style.display = 'block';
    textPasteContainer.style.display = 'none';
  });

  tabDocText.addEventListener('click', () => {
    tabDocText.classList.add('active');
    tabDocFile.classList.remove('active');
    fileUploadContainer.style.display = 'none';
    textPasteContainer.style.display = 'block';
  });

  // Scheme Document File Picker & Dropzone
  docDropzone.addEventListener('click', () => schemeFileInput.click());
  docDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    docDropzone.classList.add('drag-over');
  });
  docDropzone.addEventListener('dragleave', () => docDropzone.classList.remove('drag-over'));
  docDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    docDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      schemeFileInput.files = e.dataTransfer.files;
      updateDocFileInfo(e.dataTransfer.files[0]);
    }
  });

  schemeFileInput.addEventListener('change', () => {
    if (schemeFileInput.files.length > 0) {
      updateDocFileInfo(schemeFileInput.files[0]);
    }
  });

  function updateDocFileInfo(file) {
    docFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    docFileInfo.style.display = 'inline-flex';
  }

  // Grounding Photos Dropzone
  photoDropzone.addEventListener('click', () => groundingPhotosInput.click());
  photoDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    photoDropzone.classList.add('drag-over');
  });
  photoDropzone.addEventListener('dragleave', () => photoDropzone.classList.remove('drag-over'));
  photoDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    photoDropzone.classList.remove('drag-over');
    handlePhotos(e.dataTransfer.files);
  });

  groundingPhotosInput.addEventListener('change', () => {
    handlePhotos(groundingPhotosInput.files);
  });

  function handlePhotos(files) {
    uploadedPhotos = Array.from(files);
    photoPreviewGrid.innerHTML = '';
    uploadedPhotos.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const thumb = document.createElement('div');
        thumb.className = 'photo-thumb';
        thumb.title = file.name;
        thumb.innerHTML = `<img src="${e.target.result}" alt="${file.name}">`;
        photoPreviewGrid.appendChild(thumb);
      };
      reader.readAsDataURL(file);
    });
  }

  // Strict Validation and Generation Trigger
  btnGenerate.addEventListener('click', async () => {
    hideAlert();

    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
      showAlert("API Key is required! Please paste your Gemini or OpenAI API key. The system will not generate any output without an API key.");
      apiKeyInput.focus();
      return;
    }

    // Determine which input mode is active based on visible tab
    const isFileTabActive = tabDocFile.classList.contains('active');
    const hasFile = isFileTabActive && schemeFileInput.files && schemeFileInput.files.length > 0;
    const pastedText = (!isFileTabActive) ? schemeTextInput.value.trim() : "";

    if (!hasFile && !pastedText) {
      showAlert("Scheme document is required! Please either upload a file (in the Upload tab) or paste text (in the Paste tab).");
      return;
    }

    // Prepare FormData
    const formData = new FormData();
    formData.append("provider", providerSelect.value);
    formData.append("api_key", apiKey);
    formData.append("model_name", modelSelect.value);
    formData.append("auto_run_critic", "true");

    if (hasFile) {
      formData.append("scheme_file", schemeFileInput.files[0]);
    } else if (pastedText) {
      formData.append("scheme_text", pastedText);
      activeSchemeText = pastedText;
    }

    if (notesInput.value.trim()) {
      formData.append("additional_notes", notesInput.value.trim());
    }

    if (uploadedPhotos.length > 0) {
      uploadedPhotos.forEach(p => formData.append("grounding_photos", p));
    }

    // UI Loading state
    btnGenerate.disabled = true;
    loadingBox.classList.add('active');
    resultsSection.style.display = 'none';

    stepDoc.className = "loading-step-item active";
    stepPrompt.className = "loading-step-item";
    stepCritic.className = "loading-step-item";

    setTimeout(() => {
      stepDoc.className = "loading-step-item done";
      stepPrompt.className = "loading-step-item active";
      loadingTitle.textContent = "Directing 5-6 Cinematic Scenes & Bengali VO...";
      loadingSub.textContent = "Applying West Bengal local grounding, audio timing margins (1s start / 1-2s end), and zero-lip-sync directives.";
    }, 1500);

    setTimeout(() => {
      stepPrompt.className = "loading-step-item done";
      stepCritic.className = "loading-step-item active";
      loadingTitle.textContent = "AI Critic Agent Auditing & Refining Storyboard...";
      loadingSub.textContent = "Checking natural Bengali conversational flow, lip-sync absence, and document precision.";
    }, 4500);

    try {
      const resp = await fetch("/api/generate-plan", {
        method: "POST",
        body: formData
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || "Failed to generate video storyboard.");
      }

      currentPlan = data;
      renderStoryboard(data);
      showAlert("Video storyboard generated and refined successfully by AI Critic Agent!", true);

    } catch (err) {
      showAlert(err.message);
    } finally {
      btnGenerate.disabled = false;
      loadingBox.classList.remove('active');
    }
  });

  // Render Storyboard to DOM
  function renderStoryboard(plan) {
    displaySchemeTitle.textContent = plan.scheme_title || "West Bengal Scheme Video Plan";
    displayDuration.textContent = `⏱️ ${plan.total_duration_seconds.toFixed(1)}s Total`;
    displaySceneCount.textContent = `🎬 ${plan.scenes.length} Scenes`;
    displayGlobalGuidelines.textContent = plan.global_guidelines || "Standard WB Documentary Guidelines.";

    // Render Grounding References
    displayGroundingRefs.innerHTML = '';
    (plan.detected_grounding_references || []).forEach(ref => {
      const pill = document.createElement('span');
      pill.className = 'pill pill-wb';
      pill.innerHTML = `<span>🏛️</span> ${ref}`;
      displayGroundingRefs.appendChild(pill);
    });

    // Render AI Critic Card
    if (plan.critic_audit) {
      criticAuditCard.style.display = 'block';
      const score = plan.critic_audit.compliance_score;
      criticScoreBadge.textContent = `Score: ${score}/100 (${plan.critic_audit.passed_checks ? 'Passed' : 'Needs Review'})`;
      criticNotesText.textContent = plan.critic_audit.summary_notes;

      criticIssuesContainer.innerHTML = '';
      (plan.critic_audit.issues_found || []).forEach(iss => {
        const item = document.createElement('div');
        item.className = `critic-issue-item ${iss.severity}`;
        item.innerHTML = `<strong>[Scene ${iss.scene_number} - ${iss.category}]:</strong> ${iss.description} <br><span style="color: #34d399;">↳ Fix: ${iss.resolution}</span>`;
        criticIssuesContainer.appendChild(item);
      });
    } else {
      criticAuditCard.style.display = 'none';
    }

    // Render Scenes
    scenesContainer.innerHTML = '';
    plan.scenes.forEach((scene, index) => {
      const card = createSceneCard(scene, index);
      scenesContainer.appendChild(card);
    });

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    window.dispatchEvent(new CustomEvent('plan-updated', { detail: plan }));
  }

  // Create Individual Scene Card
  function createSceneCard(scene, index) {
    const card = document.createElement('div');
    card.className = 'scene-card';
    card.dataset.index = index;

    const startSec = scene.narration_start_second || 1.0;
    const endSec = scene.narration_end_second || 8.2;
    const durSec = scene.duration_seconds || 10.0;
    const bufferSec = (durSec - endSec).toFixed(1);

    card.innerHTML = `
      <div class="scene-header">
        <div class="scene-title-wrapper">
          <span class="scene-number-pill">Scene ${scene.scene_number}</span>
          <span class="scene-title-text">${scene.scene_title}</span>
        </div>
        <div class="scene-timing-badge">
          ⏱️ ${durSec}s Total | VO: ${startSec}s ➔ ${endSec}s (${bufferSec}s end buffer)
        </div>
      </div>

      <div class="scene-content-layout" style="display: block;">
        <div class="section-label" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
          <span>🎥 Detailed Video Scene Prompt</span>
          <button type="button" class="btn btn-secondary btn-edit-prompt" data-index="${index}" style="font-size: 0.75rem; padding: 0.25rem 0.6rem;">
            ✏️ Edit Prompt
          </button>
        </div>
        
        <div class="prompt-display-area" id="promptDisplay_${index}" style="white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); color: #cbd5e1;">${escapeHtml(scene.visual_prompt)}</div>
        
        <textarea class="editable-textarea scene-visual-input" id="promptEdit_${index}" rows="15" style="display: none; width: 100%; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; line-height: 1.5; margin-top: 0;">${scene.visual_prompt}</textarea>
        
        <!-- Store other fields hidden so they don't break JSON export -->
        <input type="hidden" class="scene-camera-input" value="${escapeHtml(scene.camera_angle)}">
        <textarea class="scene-vo-input" style="display: none;">${scene.voiceover_bengali}</textarea>
      </div>
    `;

    // Hook up Edit button
    const editBtn = card.querySelector('.btn-edit-prompt');
    const displayArea = card.querySelector(`#promptDisplay_${index}`);
    const editArea = card.querySelector(`#promptEdit_${index}`);
    
    editBtn.addEventListener('click', () => {
      if (editArea.style.display === 'none') {
        // Switch to edit mode
        editArea.style.display = 'block';
        displayArea.style.display = 'none';
        editBtn.innerHTML = '💾 Done Editing';
        editBtn.className = 'btn btn-primary btn-edit-prompt';
        editArea.focus();
      } else {
        // Switch back to view mode
        editArea.style.display = 'none';
        displayArea.style.display = 'block';
        displayArea.textContent = editArea.value;
        editBtn.innerHTML = '✏️ Edit Prompt';
        editBtn.className = 'btn btn-secondary btn-edit-prompt';
        
        // Auto-update currentPlan in memory
        if (currentPlan && currentPlan.scenes[index]) {
            currentPlan.scenes[index].visual_prompt = editArea.value;
        }
      }
    });



    return card;
  }

  // Save Scene Edits to in-memory state
  btnSaveEdits.addEventListener('click', () => {
    if (!currentPlan) return;

    const cards = document.querySelectorAll('.scene-card');
    cards.forEach((card, idx) => {
      if (currentPlan.scenes[idx]) {
        currentPlan.scenes[idx].visual_prompt = card.querySelector('.scene-visual-input').value;
        currentPlan.scenes[idx].camera_angle = card.querySelector('.scene-camera-input').value;
        currentPlan.scenes[idx].voiceover_bengali = card.querySelector('.scene-vo-input').value;
      }
    });

    showAlert("Scene edits saved successfully! You can now export the updated production plan.", true);
  });

  // Re-run Critic Agent on user edits
  btnRerunCritic.addEventListener('click', async () => {
    const key = apiKeyInput.value.trim();
    if (!key) {
      showAlert("API Key required to re-run AI Critic.");
      return;
    }
    if (!currentPlan) return;

    // Collect current edited fields first
    const cards = document.querySelectorAll('.scene-card');
    cards.forEach((card, idx) => {
      if (currentPlan.scenes[idx]) {
        currentPlan.scenes[idx].visual_prompt = card.querySelector('.scene-visual-input').value;
        currentPlan.scenes[idx].camera_angle = card.querySelector('.scene-camera-input').value;
        currentPlan.scenes[idx].voiceover_bengali = card.querySelector('.scene-vo-input').value;
      }
    });

    btnRerunCritic.disabled = true;
    btnRerunCritic.textContent = "Auditing...";

    try {
      const resp = await fetch("/api/run-critic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: providerSelect.value,
          api_key: key,
          current_plan: currentPlan,
          scheme_document_text: activeSchemeText || currentPlan.scheme_title
        })
      });

      const updatedPlan = await resp.json();
      if (!resp.ok) throw new Error(updatedPlan.detail || "Critic re-audit failed.");

      currentPlan = updatedPlan;
      renderStoryboard(updatedPlan);
      showAlert("AI Critic re-audited all scenes and applied updated refinements!", true);

    } catch (err) {
      showAlert("Critic audit error: " + err.message);
    } finally {
      btnRerunCritic.disabled = false;
      btnRerunCritic.textContent = "🔄 Re-Run Critic Audit";
    }
  });

  // Export Production Spec JSON
  btnExportJson.addEventListener('click', () => {
    if (!currentPlan) return;
    const blob = new Blob([JSON.stringify(currentPlan, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(currentPlan.scheme_title || 'scheme_video').toLowerCase().replace(/\s+/g, '_')}_spec.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Copy Video Prompts for Sora / Runway / Kling
  btnCopyPrompts.addEventListener('click', () => {
    if (!currentPlan) return;
    let text = `=== ${currentPlan.scheme_title} (Video Prompts) ===\n\n`;
    currentPlan.scenes.forEach(s => {
      text += `[Scene ${s.scene_number}: ${s.scene_title}] (${s.duration_seconds}s)\n`;
      text += `Camera: ${s.camera_angle}\n`;
      text += `Prompt: ${s.visual_prompt}\n`;
      text += `Bengali Voiceover: ${s.voiceover_bengali}\n`;
      text += `On-Screen Text: ${(s.onscreen_text || []).join(' | ')}\n\n`;
    });

    navigator.clipboard.writeText(text).then(() => {
      showAlert("All scene video prompts copied to clipboard!", true);
    }).catch(err => {
      showAlert("Failed to copy prompts: " + err.message);
    });
  });

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
