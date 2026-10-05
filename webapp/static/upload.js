/* Uploaded images and selected PDF pages. The original file stays in browser storage. */
let sourceDocument = null;
let uploadGeneration = 0;
const FILE_ACCEPT = 'image/*,application/pdf,.pdf';

async function responseError(response) {
  let detail;
  try { detail = (await response.json()).detail; } catch (_) { /* non-JSON server response */ }
  return typeof detail === 'string' ? detail : `Request failed (HTTP ${response.status}).`;
}
function showPreview(url) {
  const image = document.createElement('img');
  image.src = url;
  image.alt = sourceDocument ? 'Selected PDF page' : 'Crossword upload';
  const input = document.createElement('input');
  input.type = 'file'; input.id = 'file'; input.accept = FILE_ACCEPT;
  input.setAttribute('aria-label', 'Choose a crossword image or PDF');
  input.addEventListener('change', e => pick(e.target.files[0]));
  $('drop').replaceChildren(image, input);
}
function renderDocumentPages() {
  const panel = $('pdfPages');
  panel.classList.toggle('hidden', !sourceDocument);
  $('pdfPageChoices').replaceChildren();
  if (!sourceDocument) return;
  $('pdfFileName').textContent = `${sourceDocument.name} · ${sourceDocument.page_count} page${sourceDocument.page_count === 1 ? '' : 's'}`;
  for (const page of sourceDocument.pages) {
    const label = document.createElement('label'); label.className = 'pdf-page';
    const check = document.createElement('input'); check.type = 'checkbox';
    check.value = page.number; check.checked = sourceDocument.selected.includes(page.number);
    check.setAttribute('aria-label', `Read page ${page.number}`);
    const image = document.createElement('img'); image.src = page.preview; image.alt = `Page ${page.number}`;
    const caption = document.createElement('span'); caption.textContent = `Page ${page.number}`;
    check.addEventListener('change', () => {
      const selected = [...panel.querySelectorAll('input:checked')].map(input => Number(input.value));
      if (selected.length > 6) {
        check.checked = false;
        $('uploadStatus').textContent = 'Select up to six pages for one puzzle.';
        return;
      }
      sourceDocument.selected = selected;
      const first = sourceDocument.pages.find(page => selected.includes(page.number));
      if (first) { imageURL = first.preview; showPreview(imageURL); }
      $('uploadStatus').textContent = selected.length ? '' : 'Select at least one page to read.';
      if (typeof guidePhotoReady === 'function') guidePhotoReady(sourceDocument.name, false);
      $('read').disabled = !selected.length || !imageBlob;
      if (window.guideSave) window.guideSave();
    });
    label.append(check, image, caption); $('pdfPageChoices').appendChild(label);
  }
}
function extractionQuery(careful) {
  const query = new URLSearchParams({fast: String(!careful)});
  if (sourceDocument) query.set('pages', sourceDocument.selected.join(','));
  return query.toString();
}
function uploadForm() {
  const form = new FormData();
  form.append('file', imageBlob, imageBlob.name || (sourceDocument ? sourceDocument.name : 'puzzle.png'));
  return form;
}
async function pick(file) {
  if (!file) return;
  const generation = ++uploadGeneration;
  $('read').disabled = true;
  $('uploadStatus').textContent = 'Opening file…';
  try {
    if (file.size > 20_000_000) throw new Error('Choose a file smaller than 20 MB.');
    const signature = new TextDecoder().decode(await file.slice(0, 1024).arrayBuffer());
    const pdf = file.type === 'application/pdf' || /\.pdf$/i.test(file.name) || signature.includes('%PDF-');
    let pdfDocument = null;
    let preview;
    if (pdf) {
      const form = new FormData(); form.append('file', file, file.name);
      const response = await fetch('/api/document/preview', {method:'POST', body:form});
      if (!response.ok) throw new Error(await responseError(response));
      pdfDocument = {...await response.json(), selected:[1]};
      preview = pdfDocument.pages[0].preview;
    } else {
      preview = URL.createObjectURL(file);
      const probe = new Image(); probe.src = preview;
      try { await probe.decode(); }
      catch (_) { URL.revokeObjectURL(preview); throw new Error('Choose a readable image or PDF.'); }
    }
    if (generation !== uploadGeneration) {
      if (preview.startsWith('blob:')) URL.revokeObjectURL(preview);
      return;
    }
    if (imageURL?.startsWith('blob:')) URL.revokeObjectURL(imageURL);
    imageBlob = file; sourceDocument = pdfDocument; imageURL = preview;
    solution = null; selectedDemoPuzzle = null;
    clearSampleSelection();
    showPreview(imageURL); renderDocumentPages();
    $('read').disabled = false; $('readStatus').textContent = ''; $('uploadStatus').textContent = '';
    if (typeof guidePhotoReady === 'function') guidePhotoReady(file.name);
  } catch (error) {
    if (generation !== uploadGeneration) return;
    $('uploadStatus').textContent = error.message;
    $('read').disabled = !imageBlob || (sourceDocument && !sourceDocument.selected.length);
  }
}
function clearSampleSelection() {
  document.querySelectorAll('.chip').forEach(chip => { chip.classList.remove('sel'); chip.setAttribute('aria-pressed', 'false'); });
}
const drop = $('drop');
drop.addEventListener('dragover', event => { event.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', event => { event.preventDefault(); drop.classList.remove('over'); pick(event.dataTransfer.files[0]); });
$('file').addEventListener('change', event => pick(event.target.files[0]));
