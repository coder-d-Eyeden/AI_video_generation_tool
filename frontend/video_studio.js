/* ==========================================================================
   Video Production Studio — Gemini Omni 1.1 Flash generation, clip editing,
   timeline merge & final render. Talks to backend/video_routes.py.
   ========================================================================== */
(() => {
  const $ = (id) => document.getElementById(id);
  const bridge = () => window.studioBridge || {};
  const notify = (msg, ok = false) => (bridge().showAlert ? bridge().showAlert(msg, ok) : alert(msg));
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtSec = (s) => (s == null ? '—' : `${Number(s).toFixed(1)}s`);
  const uid = () => Math.random().toString(36).slice(2, 10);

  let plan = null;
  let clips = [];
  let timeline = [];
  const sceneJobs = {}; // scene_number -> job snapshot
  let audioModes = ['original', 'voiceover', 'mix', 'mute'];

  // ------------------------------------------------------------------ //
  // HTTP helpers
  // ------------------------------------------------------------------ //
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const detail = data && data.detail;
      throw new Error(typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : `Request failed (${res.status})`));
    }
    return data;
  }
  const postJSON = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  function pollJob(jobId, onUpdate) {
    return new Promise((resolve, reject) => {
      const tick = async () => {
        try {
          const job = await api(`/api/jobs/${jobId}`);
          if (onUpdate) onUpdate(job);
          if (job.status === 'completed') return resolve(job);
          if (job.status === 'failed') return reject(new Error(job.error || 'Job failed'));
          setTimeout(tick, 2500);
        } catch (e) { reject(e); }
      };
      tick();
    });
  }

  function progressHtml(job) {
    if (!job) return '';
    const pct = Math.max(0, Math.min(100, job.progress || 0));
    const text = job.status === 'failed' ? (job.error || 'Failed') : (job.message || job.status);
    return `<div class="job-progress ${esc(job.status)}">
      <div class="job-progress-bar"><div class="job-progress-fill" style="width:${pct}%"></div></div>
      <div class="job-progress-text" title="${esc(text)}">${esc(text)}</div></div>`;
  }

  const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',')[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });

  function omniKey() {
    return $('omniKeyInput').value.trim() || null;
  }

  async function genOptions() {
    const opts = { aspect_ratio: $('genAspect').value, resolution: $('genResolution').value, reference_images: [] };
    const key = omniKey();
    if (key) opts.api_key = key;
    if ($('useGroundingRefs').checked) {
      const photos = Array.from(bridge().getPhotos ? bridge().getPhotos() : []).slice(0, 3);
      opts.reference_images = await Promise.all(photos.map(async (f) => ({
        filename: f.name, mime_type: f.type || 'image/jpeg', base64: await fileToBase64(f),
      })));
    }
    return opts;
  }

  // ------------------------------------------------------------------ //
  // Config / key status
  // ------------------------------------------------------------------ //
  async function loadConfig() {
    try {
      const cfg = await api('/api/video/config');
      $('omniModelChip').textContent = `Model: ${cfg.model}`;
      const chip = $('omniKeyChip');
      chip.textContent = cfg.api_key_configured ? 'API key configured' : 'API key not set — paste below or in backend/config.py';
      chip.className = `meta-chip ${cfg.api_key_configured ? 'ok' : 'warn'}`;
      audioModes = cfg.audio_modes || audioModes;
      $('mergeTransition').innerHTML = (cfg.transitions || ['none', 'fade']).map((t) => `<option value="${esc(t)}" ${t === 'fade' ? 'selected' : ''}>${esc(t)}</option>`).join('');
      $('genAspect').value = cfg.default_aspect_ratio || '16:9';
      $('genResolution').value = cfg.default_resolution || '720p';
    } catch (e) {
      $('omniKeyChip').textContent = 'Backend unavailable';
    }
  }

  // ------------------------------------------------------------------ //
  // Step 1: scene generation
  // ------------------------------------------------------------------ //
  function renderSceneList() {
    const list = $('sceneGenList');
    const scenes = (plan && plan.scenes) || [];
    $('sceneGenEmpty').classList.toggle('hidden', scenes.length > 0);
    $('btnGenerateAllVideos').disabled = scenes.length === 0;
    list.innerHTML = scenes.map((s) => {
      const job = sceneJobs[s.scene_number];
      const running = job && (job.status === 'queued' || job.status === 'running');
      const hasClip = clips.some((c) => c.scene_number === s.scene_number);
      return `<div class="scene-gen-row" data-scene="${s.scene_number}">
        <div class="scene-gen-num">${s.scene_number}</div>
        <div><div class="scene-gen-title">${esc(s.scene_title)} <span class="clip-badge">${fmtSec(s.duration_seconds)}</span>${hasClip ? ' <span class="clip-badge omni">clip ready</span>' : ''}</div>
          <div class="scene-gen-sub" title="${esc(s.visual_prompt)}">${esc(s.visual_prompt)}</div></div>
        <div>${progressHtml(job)}</div>
        <button type="button" class="btn btn-secondary btn-sm" data-action="gen-scene" ${running ? 'disabled' : ''}>${hasClip ? '🔁 Regenerate' : '🎬 Generate'}</button>
      </div>`;
    }).join('');
  }

  async function generateScene(sceneNumber) {
    const scene = plan.scenes.find((s) => s.scene_number === sceneNumber);
    if (!scene) return;
    try {
      const job = await postJSON('/api/video/generate-scene', { ...(await genOptions()), scene });
      sceneJobs[sceneNumber] = job;
      renderSceneList();
      await pollJob(job.id, (j) => { sceneJobs[sceneNumber] = j; renderSceneList(); });
      notify(`Scene ${sceneNumber} video generated.`, true);
    } catch (e) {
      notify(e.message);
    } finally {
      await refreshClips();
    }
  }

  async function generateAll() {
    if (!plan) return;
    const btn = $('btnGenerateAllVideos');
    btn.disabled = true;
    try {
      const job = await postJSON('/api/video/generate-all', { ...(await genOptions()), plan });
      plan.scenes.forEach((s) => { sceneJobs[s.scene_number] = job; });
      renderSceneList();
      const done = await pollJob(job.id, (j) => { plan.scenes.forEach((s) => { sceneJobs[s.scene_number] = j; }); renderSceneList(); });
      const failures = (done.result && done.result.failures) || [];
      if (failures.length) notify(`Generated with ${failures.length} failure(s): ${failures.map((f) => `Scene ${f.scene_number}: ${f.error}`).join(' | ')}`);
      else notify('All scene videos generated. Use “Auto-build from Scenes” to assemble the timeline.', true);
    } catch (e) {
      notify(e.message);
    } finally {
      btn.disabled = false;
      await refreshClips();
    }
  }

  // ------------------------------------------------------------------ //
  // Step 2: clip library
  // ------------------------------------------------------------------ //
  async function refreshClips() {
    try { clips = await api('/api/clips'); } catch (e) { notify(e.message); }
    renderClips();
    renderSceneList();
    renderTimeline();
  }

  function sourceBadge(c) {
    if (c.source === 'omni') return '<span class="clip-badge omni">Omni</span>';
    if (c.source === 'omni-edit' || c.source === 'omni-extend') return `<span class="clip-badge edit">${esc(c.source)}</span>`;
    return `<span class="clip-badge">${esc(c.source || 'upload')}</span>`;
  }

  function renderClips() {
    $('clipLibraryEmpty').classList.toggle('hidden', clips.length > 0);
    $('clipLibrary').innerHTML = clips.map((c) => `
      <div class="clip-card" data-clip="${esc(c.id)}">
        <video src="${esc(c.url)}" ${c.thumbnail_url ? `poster="${esc(c.thumbnail_url)}"` : ''} controls preload="metadata"></video>
        <div class="clip-body">
          <div class="clip-label">${esc(c.label || c.filename)}</div>
          <div class="clip-meta">${sourceBadge(c)}<span class="clip-badge">${fmtSec(c.duration)}</span>
            ${c.width ? `<span class="clip-badge">${c.width}×${c.height}</span>` : ''}${c.has_audio ? '<span class="clip-badge">🔊</span>' : ''}</div>
          <div class="clip-actions">
            <button type="button" class="btn btn-outline-emerald btn-sm" data-action="add-tl">+ Timeline</button>
            <button type="button" class="btn btn-secondary btn-sm" data-action="show-trim">✂️ Trim</button>
            <button type="button" class="btn btn-secondary btn-sm" data-action="show-ai">✨ AI Edit</button>
            <button type="button" class="btn btn-danger btn-sm" data-action="delete">🗑️</button>
          </div>
          <div class="clip-tool hidden" data-tool="trim">
            <div class="inline"><input type="number" class="form-control" data-field="start" min="0" step="0.1" value="0" placeholder="Start (s)">
              <input type="number" class="form-control" data-field="end" min="0" step="0.1" value="${c.duration || ''}" placeholder="End (s)"></div>
            <button type="button" class="btn btn-primary btn-sm" data-action="trim">Create trimmed copy</button>
          </div>
          <div class="clip-tool hidden" data-tool="ai">
            <select class="form-select" data-field="mode"><option value="edit">Edit (restyle / change content)</option><option value="extend">Extend (continue the shot)</option></select>
            <textarea class="form-control" data-field="instruction" rows="2" placeholder="e.g. Make it golden-hour lighting, add gentle rain on the Kolkata street"></textarea>
            <div class="inline"><label class="form-label" style="margin:0;white-space:nowrap;">Extend by (s)</label><input type="number" class="form-control" data-field="extend" min="1" max="10" step="1" value="5"></div>
            <button type="button" class="btn btn-primary btn-sm" data-action="ai-edit">Run Omni edit</button>
            <div data-role="ai-progress"></div>
          </div>
        </div>
      </div>`).join('');
  }

  async function uploadClips(files) {
    for (const file of files) {
      const fd = new FormData();
      fd.append('file', file);
      try {
        await api('/api/clips/upload', { method: 'POST', body: fd });
      } catch (e) { notify(`${file.name}: ${e.message}`); }
    }
    await refreshClips();
  }

  async function handleClipAction(action, clipId, card) {
    const clip = clips.find((c) => c.id === clipId);
    if (!clip) return;
    if (action === 'add-tl') return addToTimeline(clip);
    if (action === 'show-trim' || action === 'show-ai') {
      const tool = card.querySelector(`[data-tool="${action === 'show-trim' ? 'trim' : 'ai'}"]`);
      tool.classList.toggle('hidden');
      return;
    }
    if (action === 'delete') {
      if (!confirm(`Delete "${clip.label || clip.filename}"?`)) return;
      try { await api(`/api/clips/${clipId}`, { method: 'DELETE' }); } catch (e) { notify(e.message); }
      timeline = timeline.filter((t) => t.clip_id !== clipId);
      return refreshClips();
    }
    if (action === 'trim') {
      const start = parseFloat(card.querySelector('[data-field="start"]').value) || 0;
      const endVal = card.querySelector('[data-field="end"]').value;
      try {
        await postJSON(`/api/clips/${clipId}/trim`, { start, end: endVal === '' ? null : parseFloat(endVal) });
        notify('Trimmed clip added to library.', true);
      } catch (e) { notify(e.message); }
      return refreshClips();
    }
    if (action === 'ai-edit') {
      const instruction = card.querySelector('[data-field="instruction"]').value.trim();
      if (instruction.length < 3) return notify('Describe the edit you want Omni to make.');
      const body = {
        instruction,
        mode: card.querySelector('[data-field="mode"]').value,
        extend_seconds: parseFloat(card.querySelector('[data-field="extend"]').value) || 5,
        aspect_ratio: $('genAspect').value,
        resolution: $('genResolution').value,
      };
      const key = omniKey();
      if (key) body.api_key = key;
      const box = card.querySelector('[data-role="ai-progress"]');
      const btn = card.querySelector('[data-action="ai-edit"]');
      btn.disabled = true;
      try {
        const job = await postJSON(`/api/clips/${clipId}/ai-edit`, body);
        box.innerHTML = progressHtml(job);
        await pollJob(job.id, (j) => { box.innerHTML = progressHtml(j); });
        notify('Omni edit complete — new clip added to library.', true);
        await refreshClips();
      } catch (e) {
        notify(e.message);
        btn.disabled = false;
      }
    }
  }

  // ------------------------------------------------------------------ //
  // Step 3: timeline
  // ------------------------------------------------------------------ //
  function addToTimeline(clip) {
    const scene = clip.scene || (plan && plan.scenes ? plan.scenes.find((s) => s.scene_number === clip.scene_number) : null);
    timeline.push({
      key: uid(),
      clip_id: clip.id,
      trim_start: 0,
      trim_end: null,
      speed: 1,
      audio_mode: scene && scene.voiceover_bengali ? 'voiceover' : 'original',
      narration_start: scene ? (scene.narration_start_second ?? 1) : 1,
      voiceover_text: scene ? (scene.voiceover_bengali || '') : '',
      overlay_text: scene ? (scene.onscreen_text || []).join('\n') : '',
      original_volume: 1,
      voiceover_volume: 1,
    });
    renderTimeline();
  }

  function autoBuildTimeline() {
    const sceneClips = {};
    clips.filter((c) => c.scene_number != null).forEach((c) => {
      // keep the newest clip per scene (list is newest-first)
      if (!sceneClips[c.scene_number]) sceneClips[c.scene_number] = c;
    });
    const ordered = Object.keys(sceneClips).map(Number).sort((a, b) => a - b).map((n) => sceneClips[n]);
    if (!ordered.length) return notify('No scene clips found. Generate scene videos first.');
    timeline = [];
    ordered.forEach(addToTimeline);
    if (plan && plan.scheme_title && !$('mergeTitle').value) $('mergeTitle').value = plan.scheme_title;
    notify(`Timeline built from ${ordered.length} scene clip(s).`, true);
  }

  function renderTimeline() {
    timeline = timeline.filter((t) => clips.some((c) => c.id === t.clip_id));
    $('timelineEmpty').classList.toggle('hidden', timeline.length > 0);
    const total = timeline.reduce((acc, t) => {
      const c = clips.find((x) => x.id === t.clip_id);
      const end = t.trim_end ?? (c ? c.duration : 0) ?? 0;
      return acc + Math.max(0, end - (t.trim_start || 0)) / (t.speed || 1);
    }, 0);
    $('btnMerge').textContent = timeline.length ? `🎞️ Merge & Render Final Video (~${total.toFixed(1)}s)` : '🎞️ Merge & Render Final Video';
    $('timelineList').innerHTML = timeline.map((t, i) => {
      const c = clips.find((x) => x.id === t.clip_id) || {};
      return `<div class="tl-item" data-key="${t.key}">
        ${c.thumbnail_url ? `<img class="tl-thumb" src="${esc(c.thumbnail_url)}" alt="">` : '<div class="tl-thumb"></div>'}
        <div>
          <div class="tl-head"><div class="tl-title">${i + 1}. ${esc(c.label || c.filename)} <span class="clip-badge">${fmtSec(c.duration)}</span></div>
            <div class="clip-actions">
              <button type="button" class="btn btn-secondary btn-sm" data-action="up" ${i === 0 ? 'disabled' : ''}>↑</button>
              <button type="button" class="btn btn-secondary btn-sm" data-action="down" ${i === timeline.length - 1 ? 'disabled' : ''}>↓</button>
              <button type="button" class="btn btn-danger btn-sm" data-action="remove">✕</button>
            </div></div>
          <div class="tl-fields">
            <div class="form-group"><label class="form-label">Trim start (s)</label><input type="number" class="form-control" data-field="trim_start" min="0" step="0.1" value="${t.trim_start}"></div>
            <div class="form-group"><label class="form-label">Trim end (s)</label><input type="number" class="form-control" data-field="trim_end" min="0" step="0.1" value="${t.trim_end ?? ''}" placeholder="${c.duration ?? ''}"></div>
            <div class="form-group"><label class="form-label">Speed</label><select class="form-select" data-field="speed">${[0.5, 0.75, 1, 1.25, 1.5, 2].map((v) => `<option value="${v}" ${v === Number(t.speed) ? 'selected' : ''}>${v}×</option>`).join('')}</select></div>
            <div class="form-group"><label class="form-label">Audio</label><select class="form-select" data-field="audio_mode">${audioModes.map((m) => `<option value="${m}" ${m === t.audio_mode ? 'selected' : ''}>${m}</option>`).join('')}</select></div>
            <div class="form-group"><label class="form-label">Narration at (s)</label><input type="number" class="form-control" data-field="narration_start" min="0" step="0.1" value="${t.narration_start}"></div>
          </div>
          <div class="tl-text">
            <div class="form-group"><label class="form-label">Bengali voiceover (TTS)</label><textarea class="form-control bengali" data-field="voiceover_text">${esc(t.voiceover_text)}</textarea></div>
            <div class="form-group"><label class="form-label">On-screen text (one line each)</label><textarea class="form-control bengali" data-field="overlay_text">${esc(t.overlay_text)}</textarea></div>
          </div>
        </div></div>`;
    }).join('');
  }

  function onTimelineInput(e) {
    const field = e.target.dataset.field;
    const item = e.target.closest('.tl-item');
    if (!field || !item) return;
    const t = timeline.find((x) => x.key === item.dataset.key);
    if (!t) return;
    const v = e.target.value;
    if (['trim_start', 'narration_start', 'speed'].includes(field)) t[field] = parseFloat(v) || (field === 'speed' ? 1 : 0);
    else if (field === 'trim_end') t[field] = v === '' ? null : parseFloat(v);
    else t[field] = v;
    if (e.type === 'change' && ['trim_start', 'trim_end', 'speed'].includes(field)) renderTimeline();
  }

  function onTimelineClick(e) {
    const btn = e.target.closest('[data-action]');
    const item = e.target.closest('.tl-item');
    if (!btn || !item) return;
    const i = timeline.findIndex((x) => x.key === item.dataset.key);
    const action = btn.dataset.action;
    if (action === 'remove') timeline.splice(i, 1);
    if (action === 'up' && i > 0) [timeline[i - 1], timeline[i]] = [timeline[i], timeline[i - 1]];
    if (action === 'down' && i < timeline.length - 1) [timeline[i + 1], timeline[i]] = [timeline[i], timeline[i + 1]];
    renderTimeline();
  }

  // ------------------------------------------------------------------ //
  // Background music
  // ------------------------------------------------------------------ //
  async function refreshAssets(selectId) {
    try {
      const assets = await api('/api/assets');
      const sel = $('bgMusicSelect');
      const current = selectId || sel.value;
      sel.innerHTML = '<option value="">None</option>' + assets.map((a) => `<option value="${esc(a.id)}">${esc(a.label || a.filename)}</option>`).join('');
      sel.value = assets.some((a) => a.id === current) ? current : '';
    } catch (_) { /* ignore */ }
  }

  async function uploadMusic(file) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const asset = await api('/api/assets/upload', { method: 'POST', body: fd });
      await refreshAssets(asset.id);
      notify('Background music uploaded.', true);
    } catch (e) { notify(e.message); }
  }

  // ------------------------------------------------------------------ //
  // Step 3/4: merge & finals
  // ------------------------------------------------------------------ //
  async function merge() {
    if (!timeline.length) return notify('Add at least one clip to the timeline.');
    const body = {
      title: $('mergeTitle').value.trim() || null,
      aspect_ratio: $('mergeAspect').value,
      resolution: $('mergeResolution').value,
      transition: $('mergeTransition').value,
      transition_duration: parseFloat($('mergeTransitionDur').value) || 0,
      background_music_id: $('bgMusicSelect').value || null,
      background_music_volume: Number($('bgMusicVolume').value) / 100,
      fade_in_out: $('mergeFade').checked,
      segments: timeline.map((t) => ({
        clip_id: t.clip_id,
        trim_start: t.trim_start || 0,
        trim_end: t.trim_end,
        speed: Number(t.speed) || 1,
        audio_mode: t.audio_mode,
        narration_start: t.narration_start,
        voiceover_text: (t.voiceover_text || '').trim() || null,
        overlay_text: (t.overlay_text || '').split('\n').map((s) => s.trim()).filter(Boolean),
        original_volume: t.original_volume,
        voiceover_volume: t.voiceover_volume,
      })),
    };
    const btn = $('btnMerge');
    const box = $('mergeProgress');
    btn.disabled = true;
    box.classList.remove('hidden');
    const show = (j) => { box.innerHTML = progressHtml(j); };
    try {
      const job = await postJSON('/api/video/merge', body);
      show(job);
      await pollJob(job.id, show);
      notify('Final video rendered!', true);
      await refreshFinals();
      $('finalCard').scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      notify(e.message);
    } finally {
      btn.disabled = false;
    }
  }

  async function refreshFinals() {
    let finals = [];
    try { finals = await api('/api/finals'); } catch (_) { /* ignore */ }
    $('finalEmpty').classList.toggle('hidden', finals.length > 0);
    $('finalOutput').innerHTML = finals.map((f) => `
      <div class="final-item">
        <video src="${esc(f.url)}" ${f.thumbnail_url ? `poster="${esc(f.thumbnail_url)}"` : ''} controls preload="metadata"></video>
        <div class="final-info">
          <h3>${esc(f.title || f.label || f.filename)}</h3>
          <p>${fmtSec(f.duration)} · ${f.width || '?'}×${f.height || '?'} · ${f.segments || '?'} segment(s)<br>${new Date((f.created_at || 0) * 1000).toLocaleString()}</p>
          <a class="btn btn-primary btn-sm" href="${esc(f.url)}" download="${esc(f.filename)}">⬇️ Download MP4</a>
        </div>
      </div>`).join('');
  }

  // ------------------------------------------------------------------ //
  // Wiring
  // ------------------------------------------------------------------ //
  function init() {
    $('toggleOmniKey').addEventListener('click', () => {
      const input = $('omniKeyInput');
      input.type = input.type === 'password' ? 'text' : 'password';
      $('toggleOmniKey').textContent = input.type === 'password' ? 'Show' : 'Hide';
    });
    $('btnGenerateAllVideos').addEventListener('click', generateAll);
    $('sceneGenList').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-action="gen-scene"]');
      if (btn) generateScene(Number(btn.closest('.scene-gen-row').dataset.scene));
    });

    $('btnRefreshClips').addEventListener('click', refreshClips);
    $('btnUploadClip').addEventListener('click', () => $('clipUploadInput').click());
    $('clipUploadInput').addEventListener('change', (e) => { uploadClips(Array.from(e.target.files)); e.target.value = ''; });
    $('clipLibrary').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-action]');
      const card = e.target.closest('.clip-card');
      if (btn && card) handleClipAction(btn.dataset.action, card.dataset.clip, card);
    });

    $('btnAutoTimeline').addEventListener('click', autoBuildTimeline);
    $('btnClearTimeline').addEventListener('click', () => { timeline = []; renderTimeline(); });
    $('timelineList').addEventListener('input', onTimelineInput);
    $('timelineList').addEventListener('change', onTimelineInput);
    $('timelineList').addEventListener('click', onTimelineClick);

    $('btnUploadMusic').addEventListener('click', () => $('musicUploadInput').click());
    $('musicUploadInput').addEventListener('change', (e) => { if (e.target.files[0]) uploadMusic(e.target.files[0]); e.target.value = ''; });
    $('bgMusicVolume').addEventListener('input', (e) => { $('bgMusicVolumeLabel').textContent = `${e.target.value}%`; });
    $('btnMerge').addEventListener('click', merge);

    window.addEventListener('plan-updated', (e) => {
      plan = e.detail;
      if (plan && plan.scheme_title) $('mergeTitle').value = plan.scheme_title;
      renderSceneList();
    });

    loadConfig();
    refreshClips();
    refreshAssets();
    refreshFinals();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
