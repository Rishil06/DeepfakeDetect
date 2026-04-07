import re
import os

filepath = r'c:\Users\Rishil Shah\Downloads\DEEPFAKE KACH\frontend.html'
with open(filepath, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Add #global-drop-dimmer right after <body>
if '<div id="global-drop-dimmer">' not in html:
    html = html.replace('<body>', '<body>\n\n<div id="global-drop-dimmer">\n  <div style="font-size: 3rem; margin-bottom: 12px;">👀</div>\n  <div style="font-size: 1.5rem; font-weight: 500; color: var(--text);">Drop to Analyse</div>\n</div>\n')

# 2. Add CSS for dimmer and sticky properties
css_additions = """
/* ── UI ENHANCEMENTS ── */
#global-drop-dimmer {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(240,238,235, 0.88);
  backdrop-filter: blur(8px);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  opacity: 0; pointer-events: none; transition: opacity 0.2s;
}
#global-drop-dimmer.drag-over { opacity: 1; pointer-events: all; }
body.dark #global-drop-dimmer { background: rgba(12,12,12, 0.88); }
#preview-wrap { position: relative; width: 100%; display: none; margin-top: 10px; }
#preview-wrap.show { display: block; }
#preview { width: 100%; border-radius: 9px; object-fit: cover; display: none; }
#video-preview { width: 100%; border-radius: 9px; max-height: 400px; background: #000; display: none; }
#preview-actions { position: absolute; top: 8px; right: 8px; display: flex; gap: 6px; z-index: 20; }
.btn-overlay-clear {
  background: rgba(0,0,0,0.6); color: #fff; border: none; padding: 6px 12px;
  border-radius: 6px; backdrop-filter: blur(4px); cursor: pointer; font-size: 0.75rem;
  font-family: var(--font); transition: background 0.15s;
}
.btn-overlay-clear:hover { background: rgba(0,0,0,0.8); }
"""
if '/* ── UI ENHANCEMENTS ── */' not in html:
    html = html.replace('</style>', css_additions + '\n</style>')

# Ensure verdict boxes are sticky
if 'position: sticky;' not in html:
    html = html.replace('.verdict-box {\n  background:', '.verdict-box {\n  position: sticky; top: 75px; z-index: 10;\n  background:')
    html = html.replace('.video-verdict-bar { border-radius', '.video-verdict-bar { position: sticky; top: 75px; z-index: 10; border-radius')

# 3. Remove .mode-wrap block
html = re.sub(r'<div class="mode-wrap">.*?</div>', '', html, flags=re.DOTALL)

# 4. Modify Input Panel
html = html.replace('<div id="image-panel">', '<div id="unified-panel">')
html = html.replace('<span class="panel-label">Input Image</span>', '<span class="panel-label">Input Media</span>')
html = html.replace('<div class="drop-title">Drop image here</div>', '<div class="drop-title">Drop image or video here</div>')
html = html.replace('<span class="fmt-tag">GIF</span>', '<span class="fmt-tag">GIF</span><span class="fmt-tag">MP4</span><span class="fmt-tag">MOV</span><span class="fmt-tag">WEBM</span>')

# Replace the preview img and the file input with the unified preview block
if 'id="preview-wrap"' not in html:
    html = re.sub(
        r'<img id="preview" alt="Preview"/>',
        """<div id="preview-wrap">
                  <img id="preview" alt="Preview"/>
                  <video id="video-preview" controls></video>
                  <div id="preview-actions">
                    <button class="btn-overlay-clear" onclick="event.stopPropagation(); clearAll()">✕ Clear media</button>
                  </div>
                </div>""",
        html
    )

# Update the file input
html = html.replace('accept="image/*"', 'accept="image/*,video/mp4,video/quicktime,video/webm,video/x-msvideo,video/x-matroska"')

# Remove the btn-row
html = re.sub(r'<div class="btn-row">.*?</div>', '', html, count=1, flags=re.DOTALL)

# 5. Extract `#video-results`, `#video-progress`, `#video-idle` from video panel and move them
vid_idle_match = re.search(r'(<div id="video-idle".*?</div>)', html, flags=re.DOTALL)
vid_prog_match = re.search(r'(<div class="video-progress-wrap".*?</div>\s*</div>)', html, flags=re.DOTALL)
vid_res_match = re.search(r'(<div id="video-results">.*?</div>\s*</div>)', html, flags=re.DOTALL)

# Inject them after results-content
if vid_idle_match and vid_prog_match and vid_res_match:
    injection = f"\n{vid_idle_match.group(1)}\n{vid_prog_match.group(1)}\n{vid_res_match.group(1)}\n"
    # only inject if not already there
    if 'id="video-idle"' not in html.split('<div id="results-content"')[1]:
        html = html.replace('</div>\n\n          </div>\n        </div>\n\n      </div>\n    </div>\n\n    <!-- Video Panel -->', injection + '\n          </div>\n        </div>\n\n      </div>\n    </div>\n\n    <!-- Video Panel -->')

# Now remove the entire video panel HTML
html = re.sub(r'<!-- Video Panel -->\s*<div id="video-panel".*?<!-- ─── VIEWS ─── -->', '<!-- ─── VIEWS ─── -->', html, flags=re.DOTALL)

# Now JavaScript logic
# Add global drag events
global_drag_js = """
/* ── GLOBAL DRAG & DROP ── */
var dimmer = document.getElementById('global-drop-dimmer');
var dragCounter = 0;
document.body.addEventListener('dragenter', function(e){
  e.preventDefault(); dragCounter++; dimmer.classList.add('drag-over');
});
document.body.addEventListener('dragleave', function(e){
  dragCounter--; if(dragCounter===0) dimmer.classList.remove('drag-over');
});
document.body.addEventListener('dragover', function(e){ e.preventDefault(); });
document.body.addEventListener('drop', function(e){
  e.preventDefault(); dragCounter=0; dimmer.classList.remove('drag-over');
  var f = e.dataTransfer.files[0];
  if(f) handleUnifiedDrop(f);
});

function handleUnifiedDrop(f) {
  if(f.type.startsWith('video/')) {
    imageSource = 'upload';
    onVideoFileSelect(f);
  } else if (f.type.startsWith('image/')) {
    imageSource = 'upload';
    loadFile(f);
  } else {
    showToast('Unsupported file type', 'error');
  }
}
"""

if '/* ── GLOBAL DRAG & DROP ── */' not in html:
    html = html.replace('/* ── DRAG DROP ── */', global_drag_js + '\n\n/* ── ORIGINAL LOCAL DRAG DROP ── */\n/* ── DRAG DROP ── */')

# Auto trigger analyse
html = html.replace("document.getElementById('analyse-btn').disabled=false;", 'analyseImage();')
html = html.replace("document.getElementById('video-analyse-btn').disabled=false;", 'analyseVideo();')

# Unify loadFile
html = re.sub(r"document\.getElementById\('file-input'\)\.addEventListener\('change',function\(e\)\{[\s\S]*?\}\);",
"""document.getElementById('file-input').addEventListener('change',function(e){
  var f = e.target.files[0];
  if(f) handleUnifiedDrop(f);
});
""", html)

# Modify showPreview to use preview-wrap
html = re.sub(r'function showPreview\(src\)\{.*?\}',
"""function showPreview(src){
  var vr = document.getElementById('video-results'); if(vr) vr.classList.remove('show');
  var vi = document.getElementById('video-idle'); if(vi) vi.style.display='none';
  var vprog = document.getElementById('video-progress'); if(vprog) vprog.style.display='none';
  document.getElementById('results-content').classList.remove('hidden');
  
  var w=document.getElementById('preview-wrap'); w.classList.add('show');
  var p=document.getElementById('preview'); p.src=src; p.style.display='block';
  var vpreview = document.getElementById('video-preview'); if(vpreview) vpreview.style.display='none';
  dz.classList.add('has-image');
  var o=document.getElementById('scan-overlay');o.classList.add('active');setTimeout(function(){o.classList.remove('active');},1800);
}""", html)

# Modify clearAll
html = re.sub(r'function clearAll\(\)\{.*?\}',
"""function clearAll(){
  selectedFile=null; selectedVideoFile=null; imageSource='upload';
  var w=document.getElementById('preview-wrap'); w.classList.remove('show');
  var p=document.getElementById('preview'); p.src=''; p.style.display='none';
  var vp=document.getElementById('video-preview'); if(vp){ vp.src=''; vp.style.display='none'; }
  dz.classList.remove('has-image');
  document.getElementById('file-input').value='';
  switchTab('upload'); setIdle();
  var cv = document.getElementById('conf-val'); if(cv) cv.textContent='—';
  var pv = document.getElementById('prob-val'); if(pv) pv.textContent='—';
  var cb = document.getElementById('conf-bar'); if(cb) cb.style.width='0%';
  var pb = document.getElementById('prob-bar'); if(pb) pb.style.width='0%';
  var hi=document.getElementById('heatmap-img'); if(hi){ hi.src=''; hi.classList.remove('loaded'); hi.style.display='none'; }
  var hph = document.getElementById('heatmap-placeholder'); if(hph) hph.style.display='flex';
  var hme = document.getElementById('hm-expand-btn'); if(hme) hme.style.display='none';
  var mur=document.getElementById('models-used-row'); if(mur){ mur.className=''; mur.innerHTML=''; }
  var rsk = document.getElementById('results-skeleton'); if(rsk) rsk.classList.remove('show');
  var rc = document.getElementById('results-content'); if(rc) rc.classList.remove('hidden');
  
  // video specific
  var vr = document.getElementById('video-results'); if(vr) vr.classList.remove('show');
  var vi = document.getElementById('video-idle'); if(vi) vi.style.display='block';
  var vprog = document.getElementById('video-progress'); if(vprog) vprog.style.display='none';
  if(frameChart){frameChart.destroy();frameChart=null;}
}""", html)

# Modify onVideoFileSelect
html = re.sub(r'function onVideoFileSelect\(file\)\{.*?\}',
"""function onVideoFileSelect(file){
  if(!file)return;
  selectedVideoFile=file;
  document.getElementById('results-content').classList.add('hidden');
  document.getElementById('results-skeleton').classList.remove('show');
  var w=document.getElementById('preview-wrap'); w.classList.add('show');
  document.getElementById('preview').style.display='none';
  var vp=document.getElementById('video-preview'); vp.src=URL.createObjectURL(file); vp.style.display='block';
  dz.classList.add('has-image');
  
  document.getElementById('video-results').classList.remove('show');
  document.getElementById('video-idle').style.display='block';
  analyseVideo();
}""", html)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(html)

print("HTML modifications constructed successfully.")
