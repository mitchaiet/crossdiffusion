/* Review activity and final results have separate, automatically selected views. */
let reviewActivity = [];
let reviewTab = 'thinking';
let reviewGeneration = 0;
let reviewController = null;
let reviewTimer = null;
let reviewStarted = 0;
let reviewSeconds = 0;

function selectReviewTab(tab, focus = false) {
  if (tab === 'results' && $('reviewResultsTab').disabled) return;
  reviewTab = tab;
  for (const name of ['thinking', 'results']) {
    const button = $(name === 'thinking' ? 'reviewThinkingTab' : 'reviewResultsTab');
    button.setAttribute('aria-selected', String(name === tab));
    button.tabIndex = name === tab ? 0 : -1;
    $(name === 'thinking' ? 'reviewThinking' : 'reviewResults').classList.toggle('hidden', name !== tab);
    if (focus && name === tab) button.focus();
  }
  if (window.guideSave) window.guideSave();
}
for (const [id, tab] of [['reviewThinkingTab', 'thinking'], ['reviewResultsTab', 'results']]) {
  $(id).onclick = () => selectReviewTab(tab);
  $(id).onkeydown = event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const next = event.key === 'Home' ? 'thinking' : event.key === 'End' ? 'results' : tab === 'thinking' ? 'results' : 'thinking';
      selectReviewTab(next, true);
    }
  };
}
function resetReview() {
  reviewGeneration++;
  if (reviewController) reviewController.abort();
  reviewController = null;
  clearInterval(reviewTimer); reviewTimer = null;
  reviewActivity = []; reviewSeconds = 0;
  $('repairLog').replaceChildren();
  $('reviewProgress').textContent = 'Ready to review';
  $('reviewElapsed').textContent = '';
  $('reviewSpinner').classList.add('hidden');
  $('reviewResultsTab').disabled = true;
  selectReviewTab('thinking');
}
function logRepair(message) {
  const line = document.createElement('div'); line.textContent = message;
  $('repairLog').appendChild(line); $('repairLog').scrollTop = $('repairLog').scrollHeight;
  reviewActivity.push(message);
  if (reviewActivity.length > 150) { reviewActivity.shift(); $('repairLog').firstElementChild.remove(); }
  if (window.guideSave) window.guideSave();
}
function reviewProgress(message) {
  $('reviewProgress').textContent = message;
  if (typeof guideStage !== 'undefined' && guideStage === 'rechecking') chatStatus.textContent = message;
}
function handleReviewEvent(event) {
  if (event.type === 'error') throw new Error(event.error);
  let message;
  if (event.type === 'stage') message = event.text;
  if (event.type === 'propose') message = `Round ${event.round}: checked ${event.queries} clues; found ${event.new} candidates.`;
  if (event.type === 'commit') message = `${event.slot}: using ${event.word}.`;
  if (event.type === 'reject') message = `${event.slot}: rejected ${event.word || 'the current answer'}; checking alternatives.`;
  if (event.type === 'force_fill') message = `${event.slot}: best remaining guess ${event.word}.`;
  if (event.type === 'round') message = `Round ${event.round}: ${event.commits} answers committed.`;
  if (event.type === 'judge') message = `Reviewed ${event.checked} weak answers; rejected ${event.rejected}.`;
  if (message) { reviewProgress(message); logRepair(message); }
  if (event.elapsed_s != null) {
    reviewSeconds = event.elapsed_s;
    $('reviewElapsed').textContent = `${Math.round(reviewSeconds)}s elapsed`;
  }
  if (event.grid?.length) renderGrid($('reviewLiveGrid'), event.grid);
}
function restoreReview(saved) {
  reviewActivity = []; $('repairLog').replaceChildren();
  for (const message of saved?.activity || []) logRepair(message);
  reviewSeconds = saved?.seconds || 0;
  if (reviewSeconds) $('reviewElapsed').textContent = `${Math.round(reviewSeconds)}s elapsed`;
  if (lastRevision?.final?.grid?.length) {
    renderGrid($('reviewLiveGrid'), lastRevision.final.grid);
    reviewProgress('Review complete');
    if (!reviewActivity.length) logRepair('No activity log was saved for this review.');
    selectReviewTab(saved?.tab === 'thinking' ? 'thinking' : 'results');
  }
}
$('repair').onclick = async () => {
  if (!lastResult || !llmAvailable || reviewController) return;
  const {ev: draft, p} = lastResult;
  const payload = {...p, draft_grid:draft.grid};
  resetReview(); lastRevision = null;
  const generation = reviewGeneration;
  reviewController = new AbortController();
  $('repair').disabled = true; $('repairPanel').classList.remove('hidden');
  $('repairChanges').replaceChildren(); $('repairScore').textContent = '';
  setStep(4); selectReviewTab('thinking');
  if (typeof guideBusy === 'function') guideBusy('recheck');
  renderGrid($('reviewLiveGrid'), p.grid);
  reviewProgress('Starting clue and crossing checks…');
  logRepair('The draft answers are hints. The agent will check them against the clues and crossings.');
  reviewStarted = Date.now();
  $('reviewSpinner').classList.remove('hidden');
  $('reviewElapsed').textContent = '0s elapsed';
  reviewTimer = setInterval(() => {
    reviewSeconds = (Date.now() - reviewStarted) / 1000;
    $('reviewElapsed').textContent = `${Math.floor(reviewSeconds)}s elapsed`;
  }, 1000);
  try {
    const response = await fetch('/api/solve/agent', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload), signal:reviewController.signal});
    if (!response.ok) throw new Error(await responseError(response));
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '', final = null;
    function consume(part) {
      const lines = part.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart());
      if (!lines.length) return;
      const event = JSON.parse(lines.join('\n'));
      handleReviewEvent(event);
      if (event.type === 'final') final = event;
    }
    while (true) {
      const {value, done} = await reader.read();
      if (generation !== reviewGeneration) return;
      buffer += decoder.decode(value || new Uint8Array(), {stream:!done}).replace(/\r\n/g, '\n');
      const parts = buffer.split('\n\n'); buffer = parts.pop();
      for (const part of parts) consume(part);
      if (done) { if (buffer.trim()) consume(buffer); break; }
    }
    if (!final) throw new Error('The review ended without a final grid. Try again.');
    reviewSeconds = final.elapsed_s;
    $('reviewElapsed').textContent = `${Math.round(reviewSeconds)}s elapsed`;
    reviewProgress('Review complete');
    logRepair(`Review complete. ${final.changed_cells} squares changed. Results are ready.`);
    renderRevision(final, draft, p);
    setStep(4); selectReviewTab('results');
    if (typeof guideRecheckDone === 'function') guideRecheckDone(final, draft, p);
  } catch (error) {
    if (generation !== reviewGeneration) return;
    reviewProgress('Review failed'); logRepair(error.message);
    setStep(4); selectReviewTab('thinking');
    if (typeof guideError === 'function') guideError('recheck');
  } finally {
    if (generation === reviewGeneration) {
      clearInterval(reviewTimer); reviewTimer = null; reviewController = null;
      $('reviewSpinner').classList.add('hidden');
      $('repair').disabled = !llmAvailable || !lastResult;
      if (window.guideSave) window.guideSave();
    }
  }
};

function renderRevision(final, draft, p) {
  $('repairPanel').classList.remove('hidden');
  $('reviewResultsTab').disabled = false;
  selectReviewTab('results');
  $('repairChanges').replaceChildren();
  renderGrid($('repairGrid'), final.grid, { solution: p.solution, changedFrom: draft.grid });
  const before = final.draft_score, after = final.score;
  $('repairScore').textContent = before && after
    ? `${before.letters_correct}/${before.letters_total} → ${after.letters_correct}/${after.letters_total} correct`
    : `${final.changed_cells} squares changed`;
  $('repairState').textContent = `${final.changed_cells} squares changed · ${final.llm_calls} LLM calls · ${final.elapsed_s}s${final.cost_usd == null ? '' : ' · $' + final.cost_usd.toFixed(3)}`;
  const changes = numbering(p.grid).slots.filter(s => wordIn(draft.grid, s) !== wordIn(final.grid, s));
  if (!changes.length) $('repairChanges').textContent = 'No answers changed.';
  for (const s of changes) {
    const row = document.createElement('div'); row.className = 'ans';
    const label = document.createElement('b'); label.textContent = s.id;
    const clue = document.createElement('span'); clue.textContent = (p.clues[s.dir] || {})[String(s.n)] || '';
    const words = document.createElement('span'); words.className = 'w';
    words.textContent = `${wordIn(draft.grid, s) || '—'} → ${wordIn(final.grid, s) || '—'}`;
    row.append(label, clue, words); $('repairChanges').appendChild(row);
  }
}
