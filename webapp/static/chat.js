/* A guided conversation around the existing puzzle views and solver endpoints. */
const chatLog = document.getElementById('chatHistory');
const chatActions = document.getElementById('chatActions');
const chatInput = document.getElementById('chatInput');
const chatForm = document.getElementById('chatForm');
const chatStatus = document.getElementById('chatStatus');
const chatTurns = [];
let guideStage = 'photo';
let guideHasLLM = false;
let guideHasSamples = false;
let guideWaiting = false;
let lastRevision = null;
let sampleRequested = false;
let guideWorkflow = null;

const chatActionReplies = {
  sample: 'I’ll load a sample puzzle now.',
  upload: 'I’ll open the file picker.',
  photo: 'I’ll show the uploaded file.',
  read: 'I’ll read the grid and clues, then fill a first draft.',
  puzzle: 'I’ll show the editable grid and clues.',
  fill: 'I’ll fill the grid with CrossDiffusion.',
  solve: 'I’ll read the puzzle if needed, make a first draft, then recheck it with the LLM.',
  draft: 'I’ll show the first draft.',
  recheck: 'I’ll recheck the draft against the clues and crossings.',
  review: 'I’ll show the revision.',
  new: 'I’ll start a new puzzle.',
};

function guideMessage(role, text, tone) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-message ' + role + (tone ? ' ' + tone : '');
  bubble.textContent = text;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
  if (role === 'assistant' || role === 'user') {
    chatTurns.push({role, text, tone: tone || null});
    if (chatTurns.length > 100) chatTurns.shift();
  }
  if (typeof window.guideSave === 'function') window.guideSave();
  return bubble;
}

function guideActionList() {
  switch (guideStage) {
    case 'photo': return guideHasSamples ? [['sample', 'Try a sample'], ['upload', 'Upload file', true]] : [['upload', 'Upload file', true]];
    case 'selected': return [['read', 'Read this puzzle', true], ['upload', 'Change file']];
    case 'reading': return [['puzzle', 'Watch puzzle view']];
    case 'review': return [['puzzle', 'Open puzzle'], ['fill', 'Fill first draft', true], ['upload', 'Change file']];
    case 'filling': return [['draft', 'Watch draft view']];
    case 'draft': return [['draft', 'Open first draft'], ...(guideHasLLM ? [['recheck', 'Recheck with LLM', true]] : []), ['puzzle', 'Edit puzzle']];
    case 'rechecking': return [['review', 'Watch review view']];
    case 'revised': return [['review', 'Open revision', true], ['draft', 'Compare first draft'], ['new', 'New puzzle']];
    default: return guideHasSamples ? [['sample', 'Try a sample'], ['upload', 'Upload file', true]] : [['upload', 'Upload file', true]];
  }
}

function guideRenderActions() {
  chatActions.replaceChildren();
  for (const [id, label, primary] of guideActionList()) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    if (primary) button.classList.add('primary');
    button.addEventListener('click', () => guideAction(id));
    chatActions.appendChild(button);
  }
}

function guideSet(stage, text, tone) {
  guideStage = stage;
  document.querySelector('.chat-panel').dataset.stage = stage;
  chatStatus.textContent = {
    photo: 'Waiting for a puzzle', selected: 'Photo ready', reading: 'Reading the image',
    review: 'Check the puzzle', filling: 'Making the first draft',
    draft: 'First draft ready', rechecking: 'Checking clues and crossings',
    revised: 'Revision ready',
  }[stage] || 'Waiting for a puzzle';
  if (text) guideMessage('assistant', text, tone);
  guideRenderActions();
  if (typeof window.guideSave === 'function') window.guideSave();
}

function guideAction(id) {
  if (id === 'solve') {
    guideWorkflow = 'solve';
    if (guideStage === 'selected') guideAction('read');
    else if (guideStage === 'review') guideAction('fill');
    else if (guideStage === 'draft' && guideHasLLM) guideAction('recheck');
    else if (guideStage === 'draft') {
      guideWorkflow = null;
      guideMessage('assistant', 'LLM recheck is unavailable until the server has an API key.');
    }
    else {
      guideWorkflow = null;
      guideMessage('assistant', 'Choose an image, PDF, or sample first.');
    }
  } else if (id === 'sample') {
    setStep(1);
    const first = document.querySelector('#chips .chip[data-sample-id="mini_01"]') || document.querySelector('#chips .chip');
    if (first) first.click();
    else {
      sampleRequested = true;
      guideMessage('assistant', 'Loading a sample…');
    }
  } else if (id === 'upload') {
    setStep(1);
    document.getElementById('file').click();
  } else if (id === 'photo') {
    setStep(1);
  } else if (id === 'read') {
    const button = document.getElementById('read');
    if (button.disabled) { guideWorkflow = null; guideMessage('assistant', 'Choose an image, PDF, or sample first.'); }
    else button.click();
  } else if (id === 'puzzle') {
    if (maxStep >= 2) setStep(2);
    else guideMessage('assistant', 'Read the uploaded file first to see the editable puzzle.');
  } else if (id === 'fill') {
    const button = document.getElementById('solve');
    if (button.disabled) { guideWorkflow = null; guideMessage('assistant', 'Read and check the puzzle first.'); }
    else button.click();
  } else if (id === 'draft') {
    if (maxStep >= 3) setStep(3);
    else guideMessage('assistant', 'Fill the puzzle first to see a draft.');
  } else if (id === 'recheck') {
    const button = document.getElementById('repair');
    if (!guideHasLLM) { guideWorkflow = null; guideMessage('assistant', 'LLM recheck is unavailable until the server has an API key.'); }
    else if (button.disabled) { guideWorkflow = null; guideMessage('assistant', 'Wait for the first draft to finish.'); }
    else button.click();
  } else if (id === 'review') {
    if (maxStep >= 4) setStep(4);
    else guideMessage('assistant', 'Run the LLM recheck first to see a revision.');
  } else if (id === 'new') {
    if (typeof window.guideNewSession === 'function') window.guideNewSession();
    else window.location.reload();
  }
}

function guideViewChanged() {
  guideRenderActions();
  if (typeof window.guideSave === 'function') window.guideSave();
}
function guideDemosLoaded(count) {
  guideHasSamples = count > 0;
  guideRenderActions();
  if (sampleRequested) {
    sampleRequested = false;
    const first = document.querySelector('#chips .chip[data-sample-id="mini_01"]') || document.querySelector('#chips .chip');
    if (first) first.click();
    else guideMessage('assistant', 'No bundled samples are available. Upload an image or PDF.');
  }
}
function guideProgress(text) {
  if (guideStage === 'reading') chatStatus.textContent = text.slice(0, 90);
}
function guideConfig(config) {
  guideHasLLM = Boolean(config.has_key);
  guideRenderActions();
}
function guidePhotoReady(name, announce = true) {
  resetReview();
  guideWorkflow = null;
  lastRevision = null;
  lastResult = null;
  P = {title:'Puzzle', grid:[], clues:{across:{}, down:{}}};
  maxStep = 1;
  setStep(1);
  guideSet('selected', announce ? 'Loaded ' + name + '. Type “read” to extract and fill it, or “solve” to include an LLM review.' : null);
}
function guideBusy(kind) {
  if (kind === 'read' || kind === 'fill') resetReview();
  if (kind === 'read') {
    lastRevision = null;
    guideSet('reading');
  }
  if (kind === 'fill') {
    lastRevision = null;
    guideSet('filling');
  }
  if (kind === 'recheck') guideSet('rechecking');
}
function guideReadDone(entries, missing, problems = []) {
  if (missing || problems.length) guideWorkflow = null;
  const continuing = Boolean(guideWorkflow);
  guideSet('review', 'Read ' + entries + ' entries. ' +
    (missing ? missing + ' missing clue' + (missing === 1 ? '' : 's') + ' marked in red. ' : '') +
    (continuing ? 'Filling the first draft…' : 'Check the grid and clues, then fill the draft.'));
  if (continuing) guideAction('fill');
}
function guideFillDone(result, puzzle) {
  const score = result.score;
  const rechecking = guideWorkflow === 'solve' && guideHasLLM;
  guideSet('draft', 'The first draft is ready. ' +
    (score ? score.letters_correct + '/' + score.letters_total + ' letters match the answer key. ' : 'No answer key is available for scoring. ') +
    (rechecking ? 'Checking clues and crossings…'
      : guideHasLLM ? 'Type “recheck” for an LLM review.' : 'Inspect the grid and answers in the draft view.'));
  guideWorkflow = null;
  if (rechecking) guideAction('recheck');
}
function guideRecheckDone(final, draft, puzzle) {
  guideWorkflow = null;
  lastRevision = {final, draft, puzzle};
  const score = final.score;
  guideSet('revised', 'The LLM revision is ready. It changed ' + final.changed_cells + ' squares. ' +
    (score ? score.letters_correct + '/' + score.letters_total + ' letters match the answer key. ' : 'No answer key is available for scoring. ') +
    'Changed answers are listed below the grid.');
}
function guideError(kind) {
  guideWorkflow = null;
  if (kind === 'read') guideSet('selected', 'Reading failed. See the puzzle view for details; retry or choose another photo.', 'error');
  if (kind === 'fill') guideSet('review', 'Draft failed. See the draft view for details, then retry.', 'error');
  if (kind === 'recheck') guideSet('draft', 'LLM review failed. See the review view for details; retry from the draft.', 'error');
}

function guideContext() {
  const data = {title: P.title, size: P.grid.length ? P.grid[0].length + '×' + P.grid.length : null,
    photo_selected: Boolean(imageBlob), sample_selected: Boolean(selectedDemoPuzzle), llm_available: guideHasLLM};
  if (P.grid.length) {
    const slots = numbering(P.grid).slots;
    data.entries = slots.length;
    data.missing_clues = slots.filter(s => P.clues[s.dir][s.n] == null).map(s => s.id).slice(0, 15);
    data.clues = slots.map(s => ({id: s.id, clue: P.clues[s.dir][s.n] || ''})).slice(0, 60);
  }
  if (lastResult) {
    data.draft_score = lastResult.ev.score || null;
    data.draft_grid = lastResult.ev.grid;
  }
  if (lastRevision) {
    data.revised_score = lastRevision.final.score || null;
    data.revised_grid = lastRevision.final.grid;
    data.changed_cells = lastRevision.final.changed_cells;
    data.changed_answers = numbering(lastRevision.puzzle.grid).slots
      .map(s => ({id:s.id, clue:lastRevision.puzzle.clues[s.dir][s.n] || '',
        before:wordIn(lastRevision.draft.grid, s), after:wordIn(lastRevision.final.grid, s)}))
      .filter(s => s.before !== s.after).slice(0, 30);
  }
  return data;
}

function guideLocalIntent(text) {
  const t = text.trim().toLowerCase().replace(/[.!?]+$/, '');
  if (/^(try (a )?sample|use (a )?sample|sample)$/.test(t)) return 'sample';
  if (/^(upload|choose|change|add) (a |the )?(photo|image|picture|file|pdf)$/.test(t)) return 'upload';
  if (/^(read|scan|extract)( (it|the photo|the pdf|the file|this puzzle|the puzzle( i (loaded|uploaded|selected))?|the (image|photo|pdf|file)( i (loaded|uploaded|selected))?))?$/.test(t)) return 'read';
  if (/^(solve|finish|complete)( (it|this puzzle|the puzzle))?$/.test(t)) return 'solve';
  if (/^(fill|make (a |the )?draft|run (the )?solver)( (it|the puzzle))?$/.test(t)) return 'fill';
  if (/^(recheck|check (the )?answers|run (the )?llm|run (the )?agent)( (it|the puzzle))?$/.test(t)) return 'recheck';
  if (/^(show|open|view) (the )?(photo|image)$/.test(t)) return 'photo';
  if (/^(show|open|view|edit) (the )?(puzzle|grid|clues)$/.test(t)) return 'puzzle';
  if (/^(show|open|view) (the )?(draft|first draft)$/.test(t)) return 'draft';
  if (/^(show|open|view) (the )?(review|revision|result)$/.test(t)) return 'review';
  if (/^(start over|new puzzle)$/.test(t)) return 'new';
  if (/^(next|continue|go on)$/.test(t)) return {selected:'read', review:'fill', draft:'recheck', revised:'review'}[guideStage] || null;
  return null;
}

function guideChatAction(action) {
  guideMessage('assistant', chatActionReplies[action] || 'I’ll do that now.');
  if (action === 'read') guideWorkflow = 'readFill';
  guideAction(action);
}

async function guideAsk(text) {
  const stageAtSend = guideStage;
  const previous = chatTurns.slice(0, -1).filter(turn => turn.tone !== 'chat-pending').slice(-24);
  const pending = guideMessage('assistant', 'Thinking…', 'chat-pending');
  const pendingTurn = chatTurns[chatTurns.length - 1];
  guideWaiting = true;
  document.getElementById('chatSend').disabled = true;
  try {
    const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text, stage:guideStage, history:previous, context:guideContext()})});
    const data = await response.json();
    if (!response.ok) throw new Error(response.status === 503 ? 'Nebius is not configured on this server yet.' : 'I couldn’t answer that right now. The step buttons still work.');
    pending.classList.remove('chat-pending');
    const action = data.action && stageAtSend === guideStage ? data.action : null;
    pending.textContent = action ? chatActionReplies[action] : String(data.reply).replace(/\*\*/g, '');
    pendingTurn.text = pending.textContent;
    pendingTurn.tone = null;
    guideHasLLM = true;
    llmAvailable = true;
    document.getElementById('repair').disabled = !lastResult;
    guideRenderActions();
    if (typeof window.guideSave === 'function') window.guideSave();
    if (action) {
      if (action === 'read') guideWorkflow = 'readFill';
      guideAction(action);
    }
  } catch (error) {
    pending.classList.remove('chat-pending');
    pending.classList.add('error');
    pending.textContent = error.message === 'Nebius is not configured on this server yet.'
      ? error.message : 'I couldn’t answer that right now. The step buttons still work.';
    pendingTurn.text = pending.textContent;
    pendingTurn.tone = 'error';
    if (typeof window.guideSave === 'function') window.guideSave();
  } finally {
    guideWaiting = false;
    document.getElementById('chatSend').disabled = false;
    chatLog.scrollTop = chatLog.scrollHeight;
  }
}

chatForm.addEventListener('submit', event => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text || guideWaiting) return;
  chatInput.value = '';
  guideMessage('user', text);
  const action = guideLocalIntent(text);
  if (action) guideChatAction(action);
  else guideAsk(text);
});
chatInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});
window.addEventListener('resize', () => { chatLog.scrollTop = chatLog.scrollHeight; });
guideSet('photo', 'Upload a puzzle to start.');
if (llmAvailable) guideConfig({has_key:true});
