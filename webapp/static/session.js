/* Keep each puzzle, its views, and its conversation in this browser. */
(() => {
  const STORE = 'sessions';
  const ACTIVE = 'crossdiffusion-active-session';
  const picker = document.getElementById('chatSessionPicker');
  const status = document.getElementById('chatSaveStatus');
  const send = document.getElementById('chatSend');
  const input = document.getElementById('chatInput');
  const actions = document.getElementById('chatActions');
  const newButton = document.getElementById('chatNew');
  let db;
  let ready = false;
  let restoring = false;
  let saveTimer;
  let saveGeneration = 0;
  let saveChain = Promise.resolve();
  const makeId = () => Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  let sessionId;
  try {
    sessionId = localStorage.getItem(ACTIVE) || makeId();
    localStorage.setItem(ACTIVE, sessionId);
  } catch (_) { sessionId = makeId(); }

  send.disabled = true;
  input.disabled = true;
  picker.disabled = true;
  newButton.disabled = true;
  actions.style.pointerEvents = 'none';

  function openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('crossdiffusion-conversations', 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE, {keyPath:'id'});
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  function read(id) {
    return new Promise((resolve, reject) => {
      const request = db.transaction(STORE, 'readonly').objectStore(STORE).get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  function all() {
    return new Promise((resolve, reject) => {
      const request = db.transaction(STORE, 'readonly').objectStore(STORE).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  function write(record) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(record);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  }
  function snapshot() {
    const visible = document.querySelector('.page.active');
    return {
      version: 1, stage: guideStage, step: Number(visible?.id.slice(1)) || 1, maxStep,
      turns: chatTurns.filter(turn => turn.tone !== 'chat-pending').slice(-100),
      puzzle: P, solution, selectedDemoPuzzle, lastResult, lastRevision,
      sourceDocument, review: {activity:reviewActivity, tab:reviewTab, seconds:reviewSeconds},
      imageBlob, imageURL: imageURL?.startsWith('blob:') ? null : imageURL,
      settings: {careful: $('careful').checked, steps: $('steps').value,
        pace: $('pace').value, batchMode: $('batchMode').value},
    };
  }
  function titleOf(state) {
    if (state.puzzle?.grid?.length) return state.puzzle.title || 'Puzzle';
    if (state.imageBlob?.name) return state.imageBlob.name;
    if (state.imageURL) return state.imageURL.split('/').pop();
    return 'New puzzle';
  }
  async function refreshPicker() {
    const records = (await all()).sort((a, b) => b.updated - a.updated);
    picker.replaceChildren();
    for (const record of records) {
      const option = document.createElement('option');
      option.value = record.id;
      option.textContent = `${record.title} · ${new Date(record.updated).toLocaleDateString()}`;
      picker.appendChild(option);
    }
    picker.value = sessionId;
  }
  function saveNow() {
    if (!ready || restoring || !db) return saveChain;
    clearTimeout(saveTimer);
    const state = snapshot();
    const generation = saveGeneration;
    const record = {id: sessionId, updated: Date.now(), title: titleOf(state), state};
    saveChain = saveChain.catch(() => {}).then(async () => {
      try {
        await write(record);
        if (generation === saveGeneration) status.textContent = 'Saved on this browser';
      } catch (error) {
        if (state.imageBlob) {
          state.imageBlob = null; state.sourceDocument = null; state.imageURL = null;
          try {
            await write(record);
            status.textContent = 'Saved without upload (browser storage full)';
          } catch (_) {
            status.textContent = 'Could not save this conversation';
            return;
          }
        } else {
          status.textContent = 'Could not save this conversation';
          return;
        }
      }
      await refreshPicker();
    });
    return saveChain;
  }
  window.guideSave = () => {
    if (!ready || restoring) return;
    saveGeneration++;
    status.textContent = 'Saving…';
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveNow, 120);
  };
  window.guideNewSession = async () => {
    await saveNow();
    try { localStorage.setItem(ACTIVE, makeId()); } catch (_) { /* session still restarts */ }
    window.location.reload();
  };

  function restoreDraft(saved) {
    if (!saved?.ev?.grid?.length || !saved?.p?.grid?.length) return false;
    lastResult = saved;
    const {ev, p} = saved;
    renderGrid($('grid'), ev.grid, {solution:p.solution, conf:ev.confidence});
    renderAnswers(ev, p);
    const count = ev.snapshots?.length || 1;
    $('mStep').textContent = `${count} / ${count}`;
    $('mFill').textContent = pct(fillPct(ev.grid));
    $('bar').style.width = pct(fillPct(ev.grid));
    $('mGpu').textContent = `${ev.elapsed_s}s`;
    if (ev.score) {
      $('mScoreL').textContent = 'Letters correct';
      $('mScore').textContent = `${ev.score.letters_correct}/${ev.score.letters_total}`;
      setState(`${pct(ev.score.letter_accuracy)} of letters correct`, 'done');
    } else {
      $('mScoreL').textContent = 'Mean confidence';
      const values = Object.values(ev.confidence || {});
      $('mScore').textContent = pct(values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);
      setState('filled', 'done');
    }
    $('solve').disabled = false;
    $('replay').disabled = false;
    $('again').disabled = false;
    $('repair').disabled = !llmAvailable;
    return true;
  }
  function restoreSession(state) {
    restoring = true;
    try {
      if (state.version !== 1) return;
      chatTurns.length = 0;
      chatLog.replaceChildren();
      for (const turn of state.turns || []) {
        if ((turn.role === 'user' || turn.role === 'assistant') && typeof turn.text === 'string') {
          guideMessage(turn.role, turn.text, turn.tone);
        }
      }
      P = state.puzzle?.grid?.length ? state.puzzle : {title:'Puzzle', grid:[], clues:{across:{}, down:{}}};
      solution = state.solution || null;
      selectedDemoPuzzle = state.selectedDemoPuzzle || null;
      imageBlob = state.imageBlob || null;
      sourceDocument = state.sourceDocument || null;
      imageURL = state.imageURL || (imageBlob ? URL.createObjectURL(imageBlob) : null);
      if (imageURL) {
        showPreview(imageURL);
        $('origImg').src = imageURL;
        $('read').disabled = !imageBlob;
      }
      renderDocumentPages();
      if (sourceDocument && !sourceDocument.selected.length) $('read').disabled = true;
      const settings = state.settings || {};
      if (typeof settings.careful === 'boolean') $('careful').checked = settings.careful;
      for (const id of ['steps', 'pace', 'batchMode']) {
        if (settings[id] != null) $(id).value = settings[id];
      }
      let availableStep = 1;
      if (P.grid.length) {
        revealedRows = Infinity;
        renderPaper();
        $('readMeta').textContent = 'Restored from this browser';
        $('solve').disabled = false;
        availableStep = 2;
      }
      if (restoreDraft(state.lastResult)) availableStep = 3;
      lastRevision = state.lastRevision || null;
      if (lastRevision?.final?.grid?.length && lastResult) {
        renderRevision(lastRevision.final, lastRevision.draft, lastRevision.puzzle);
        availableStep = 4;
      } else lastRevision = null;
      restoreReview(state.review);
      const interrupted = {reading:'selected', filling:'review', rechecking:'draft'};
      let stage = interrupted[state.stage] || state.stage;
      if (stage === 'selected' && !imageBlob) stage = 'photo';
      if (['review', 'filling'].includes(stage) && availableStep < 2) stage = imageBlob ? 'selected' : 'photo';
      if (['draft', 'rechecking'].includes(stage) && availableStep < 3) stage = availableStep >= 2 ? 'review' : 'photo';
      if (stage === 'revised' && availableStep < 4) stage = availableStep >= 3 ? 'draft' : 'photo';
      maxStep = availableStep;
      guideSet(stage);
      setStep(Math.min(Math.max(Number(state.step) || 1, 1), availableStep));
      if (interrupted[state.stage]) guideMessage('assistant', 'Reload interrupted the previous step. Retry to continue.');
      if (!chatTurns.length) guideMessage('assistant', 'Upload a puzzle to start.');
    } finally { restoring = false; }
  }
  async function start() {
    try {
      db = await openDatabase();
      const saved = await read(sessionId);
      if (saved?.state) restoreSession(saved.state);
      ready = true;
      await saveNow();
    } catch (error) {
      status.textContent = 'Saving is unavailable in this browser';
      picker.disabled = true;
    } finally {
      send.disabled = false;
      input.disabled = false;
      if (db) picker.disabled = false;
      newButton.disabled = false;
      actions.style.pointerEvents = '';
    }
  }
  newButton.addEventListener('click', () => guideAction('new'));
  picker.addEventListener('change', async () => {
    const selected = picker.value;
    await saveNow();
    try { localStorage.setItem(ACTIVE, selected); } catch (_) { return; }
    window.location.reload();
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) saveNow(); });
  start();
})();
