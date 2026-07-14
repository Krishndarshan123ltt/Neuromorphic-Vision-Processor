import os
import sys
import io
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import (Flask, request,
                   jsonify, Response,
                   redirect)
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from app import (NeuromorphicSNN,
                 predict_image,
                 load_mnist,
                 evaluate_model)
from metrics import run_metrics

os.makedirs('results', exist_ok=True)

app    = Flask(__name__)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu')
model  = None

state = {
    'n_predictions': 0,
    'accuracy':      0.0,
    'history':       [],
}


def init_model():
    global model
    print("🧠 Loading SNN model...")
    model = NeuromorphicSNN().to(device)
    wp    = 'results/best_weights.pth'
    if os.path.exists(wp):
        model.load_state_dict(
            torch.load(wp, map_location=device))
        print("✅ Weights loaded!")
    else:
        print("⚠️  No weights — train first!")
    try:
        with open(
                'results/last_accuracy.txt') as f:
            state['accuracy'] = float(
                f.read().strip())
    except:
        state['accuracy'] = 0.0


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                bbox_inches='tight',
                facecolor='#1a1a2e')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def arr_to_b64(arr):
    fig, ax = plt.subplots(
        figsize=(3, 3), facecolor='#1a1a2e')
    ax.imshow(arr, cmap='gray',
              interpolation='nearest')
    ax.axis('off')
    return fig_to_b64(fig)


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080818;color:#e0e0e0;
     font-family:'Rajdhani',sans-serif}
.nav{background:rgba(15,15,35,0.95);
     padding:14px 28px;display:flex;
     align-items:center;
     justify-content:space-between;
     border-bottom:1px solid rgba(233,30,99,0.3);
     position:sticky;top:0;z-index:100}
.nav h1{color:#E91E63;
        font-family:'Orbitron',sans-serif;
        font-size:1rem;letter-spacing:2px}
.nav a{color:#90CAF9;text-decoration:none;
       margin-left:18px;font-size:.9rem}
.nav a:hover{color:#E91E63}
.wrap{max-width:1200px;margin:28px auto;
      padding:0 18px}
.card{background:rgba(20,20,45,0.8);
      border:1px solid rgba(255,255,255,0.08);
      border-radius:16px;padding:24px;
      margin-bottom:20px}
.card h2{color:#E91E63;margin-bottom:15px;
         font-family:'Orbitron',sans-serif;
         font-size:.9rem;letter-spacing:2px}
.btn{background:linear-gradient(
       135deg,#E91E63,#C2185B);
     color:#fff;border:none;
     padding:10px 24px;border-radius:8px;
     cursor:pointer;font-size:.9rem;
     letter-spacing:1px;text-decoration:none;
     display:inline-block;transition:all .2s}
.btn:hover{transform:translateY(-2px)}
.btn-b{background:linear-gradient(
         135deg,#1565C0,#0D47A1)}
.btn-g{background:linear-gradient(
         135deg,#2E7D32,#1B5E20)}
.g2{display:grid;
    grid-template-columns:1fr 1fr;gap:20px}
.g3{display:grid;
    grid-template-columns:1fr 1fr 1fr;gap:20px}
.mbox{background:rgba(10,10,25,0.8);
      border:1px solid rgba(255,255,255,0.06);
      border-radius:12px;padding:20px;
      text-align:center}
.mv{font-size:2rem;font-weight:bold;
    color:#E91E63;
    font-family:'Orbitron',sans-serif}
.ml{color:#90CAF9;font-size:.82rem;
    margin-top:5px;letter-spacing:1px}
.tag{display:inline-block;padding:3px 10px;
     border-radius:4px;font-size:.75rem;
     border:1px solid #E91E63;color:#E91E63;
     background:rgba(233,30,99,0.1)}
.tag-g{border-color:#4CAF50;color:#4CAF50;
       background:rgba(76,175,80,0.1)}
table{width:100%;border-collapse:collapse}
th{background:rgba(233,30,99,0.1);
   color:#E91E63;padding:10px;
   text-align:left;font-size:.85rem}
td{padding:8px 10px;
   border-bottom:1px solid rgba(255,255,255,0.05);
   font-size:.88rem}
.sbar{display:flex;align-items:flex-end;
      gap:3px;height:80px;margin-top:10px}
.sc{flex:1;display:flex;
    flex-direction:column;align-items:center}
.sf{width:100%;border-radius:3px 3px 0 0;
    min-height:2px}
.sl{font-size:.62rem;color:#90CAF9;
    margin-top:3px}
img.ri{border-radius:8px;max-width:100%;
       border:1px solid rgba(255,255,255,0.1)}
.upload{border:2px dashed rgba(255,255,255,0.1);
        border-radius:12px;padding:30px;
        text-align:center;cursor:pointer;
        transition:all .3s}
.upload:hover{border-color:#E91E63}
#draw-canvas{
  border:2px solid rgba(233,30,99,0.5);
  border-radius:12px;cursor:crosshair;
  background:#000;display:block}
#draw-pred{font-size:7rem;font-weight:900;
           font-family:'Orbitron',sans-serif;
           color:#E91E63;text-align:center}
#draw-conf{color:#90CAF9;font-size:1rem;
           margin-top:8px;text-align:center}
#draw-sbar{display:flex;align-items:flex-end;
           gap:3px;height:80px;width:100%;
           margin-top:15px}
@media(max-width:768px){
  .g2,.g3{grid-template-columns:1fr}}
</style>
"""

NAV = """
<nav class="nav">
  <h1>🧠 NEUROMORPHIC VISION</h1>
  <div>
    <a href="/">🏠 Home</a>
    <a href="/draw">✏️ Draw</a>
    <a href="/upload">📤 Upload</a>
    <a href="/webcam">📷 Webcam</a>
    <a href="/spikes">⚡ Spikes</a>
    <a href="/metrics">📊 Metrics</a>
    <a href="/history">📋 History</a>
  </div>
</nav>
"""


@app.route('/')
def index():
    trained = os.path.exists(
        'results/best_weights.pth')
    acc = state['accuracy']
    html = f"""<!DOCTYPE html><html><head>
    <title>Neuromorphic Vision</title>
    {STYLE}</head><body>{NAV}
    <div class="wrap">
    <div class="g3" style="margin-bottom:22px">
      <div class="mbox">
        <div class="mv">
          {'✅' if trained else '⚠️'}
        </div>
        <div class="ml">MODEL STATUS</div>
        <div style="margin-top:10px">
          <span class="tag
            {'tag-g' if trained else ''}">
            {'TRAINED' if trained
             else 'NEEDS TRAINING'}
          </span>
        </div>
      </div>
      <div class="mbox">
        <div class="mv">{acc:.1f}%</div>
        <div class="ml">TEST ACCURACY</div>
        <div style="margin-top:10px">
          <span class="tag
            {'tag-g' if acc>=92 else ''}">
            {'✅ TARGET MET' if acc>=92
             else '⚠️ TRAIN MORE'}
          </span>
        </div>
      </div>
      <div class="mbox">
        <div class="mv">
          {state['n_predictions']}
        </div>
        <div class="ml">TOTAL PREDICTIONS</div>
      </div>
    </div>
    <div class="card">
      <h2>🏗️ NETWORK ARCHITECTURE</h2>
      <div class="g2">
        <div>
          <p style="color:#90CAF9;
                    margin-bottom:12px">
            SNN PIPELINE
          </p>
          <div style="font-family:monospace;
                      color:#A5D6A7;
                      font-size:.83rem;
                      line-height:2.2">
            📷 Input (28×28)<br>
            ↓ Flatten → 784 neurons<br>
            🔵 FC Layer 1: 784→1000<br>
            ⚡ LIF Neurons (β=0.95)<br>
            ↓<br>
            🔴 FC Layer 2: 1000→10<br>
            ⚡ LIF Neurons<br>
            ↓ 25 timesteps<br>
            🎯 Prediction (0–9)
          </div>
        </div>
        <div>
          <table>
            <tr><th>Parameter</th>
                <th>Value</th></tr>
            <tr><td>Framework</td>
                <td><span class="tag">
                  snnTorch+PyTorch
                </span></td></tr>
            <tr><td>Neuron</td>
                <td><span class="tag">
                  LIF (Leaky)
                </span></td></tr>
            <tr><td>Learning</td>
                <td><span class="tag">
                  Surrogate Gradient
                </span></td></tr>
            <tr><td>Timesteps</td>
                <td><span class="tag">
                  25
                </span></td></tr>
            <tr><td>Beta</td>
                <td><span class="tag">
                  0.95
                </span></td></tr>
            <tr><td>Dataset</td>
                <td><span class="tag">
                  MNIST 60K
                </span></td></tr>
            <tr><td>Accuracy</td>
                <td><span class="tag tag-g">
                  {acc:.1f}%
                </span></td></tr>
          </table>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>🚀 QUICK ACTIONS</h2>
      <div style="display:flex;gap:12px;
                  flex-wrap:wrap">
        <a href="/draw" class="btn">
          ✏️ Draw</a>
        <a href="/test_sample" class="btn btn-b">
          🎲 Random Test</a>
        <a href="/upload" class="btn btn-g">
          📤 Upload</a>
        <a href="/webcam" class="btn"
           style="background:linear-gradient(
             135deg,#6A1B9A,#4A148C)">
           📷 Webcam</a>
        <a href="/metrics" class="btn"
           style="background:linear-gradient(
             135deg,#E65100,#BF360C)">
           📊 Metrics</a>
        <a href="/history" class="btn"
           style="background:linear-gradient(
             135deg,#00695C,#004D40)">
           📋 History</a>
      </div>
    </div>
    </div></body></html>"""
    return html


@app.route('/draw')
def draw_page():
    html = f"""<!DOCTYPE html><html><head>
    <title>Draw</title>{STYLE}</head>
    <body>{NAV}<div class="wrap">
    <div class="g2">
      <div class="card">
        <h2>✏️ DRAW A DIGIT</h2>
        <canvas id="draw-canvas"
                width="280" height="280">
        </canvas>
        <div style="display:flex;gap:10px;
                    margin-top:12px;
                    flex-wrap:wrap">
          <button class="btn"
                  onclick="predictDrawing()">
            🧠 Predict
          </button>
          <button class="btn btn-b"
                  onclick="clearCanvas()">
            🗑️ Clear
          </button>
          <select id="brushSize"
                  style="background:#1a1a3e;
                         color:white;
                         border:1px solid #E91E63;
                         padding:8px;
                         border-radius:8px">
            <option value="15">Small</option>
            <option value="22" selected>
              Medium</option>
            <option value="30">Large</option>
          </select>
        </div>
        <p style="color:#555;font-size:.78rem;
                  margin-top:10px">
          💡 Draw BIG and THICK!
             White digit on black canvas!
        </p>
      </div>
      <div class="card">
        <h2>🎯 RESULT</h2>
        <div id="draw-pred">—</div>
        <div id="draw-conf">
          Draw digit → Click Predict!
        </div>
        <div id="draw-sbar"></div>
        <div id="draw-extra"
             style="margin-top:12px;
                    text-align:center">
        </div>
      </div>
    </div>
    <script>
    const canvas = document.getElementById(
      'draw-canvas');
    const ctx    = canvas.getContext('2d');
    let drawing  = false;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0,
      canvas.width, canvas.height);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = 22;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';

    document.getElementById('brushSize')
      .addEventListener('change', function(){{
        ctx.lineWidth = parseInt(this.value);
      }});

    canvas.addEventListener('mousedown', e=>{{
      drawing = true;
      const r = canvas.getBoundingClientRect();
      ctx.beginPath();
      ctx.moveTo(e.clientX-r.left,
                 e.clientY-r.top);
    }});
    canvas.addEventListener('mousemove', e=>{{
      if(!drawing) return;
      const r = canvas.getBoundingClientRect();
      ctx.lineTo(e.clientX-r.left,
                 e.clientY-r.top);
      ctx.stroke();
    }});
    canvas.addEventListener('mouseup',
      ()=> drawing=false);
    canvas.addEventListener('mouseleave',
      ()=> drawing=false);
    canvas.addEventListener('touchstart',e=>{{
      e.preventDefault(); drawing=true;
      const r=canvas.getBoundingClientRect();
      const t=e.touches[0];
      ctx.beginPath();
      ctx.moveTo(t.clientX-r.left,
                 t.clientY-r.top);
    }});
    canvas.addEventListener('touchmove',e=>{{
      e.preventDefault();
      if(!drawing) return;
      const r=canvas.getBoundingClientRect();
      const t=e.touches[0];
      ctx.lineTo(t.clientX-r.left,
                 t.clientY-r.top);
      ctx.stroke();
    }});
    canvas.addEventListener('touchend',
      ()=> drawing=false);

    function clearCanvas(){{
      ctx.fillStyle='#000000';
      ctx.fillRect(0,0,
        canvas.width,canvas.height);
      document.getElementById('draw-pred')
        .textContent='—';
      document.getElementById('draw-conf')
        .textContent='Draw digit → Click Predict!';
      document.getElementById('draw-sbar')
        .innerHTML='';
      document.getElementById('draw-extra')
        .innerHTML='';
    }}

    async function predictDrawing(){{
      const imageData=canvas.toDataURL('image/png');
      document.getElementById('draw-pred')
        .textContent='...';
      document.getElementById('draw-conf')
        .textContent='Processing...';
      try{{
        const res=await fetch('/predict_drawing',{{
          method:'POST',
          headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{image:imageData}})
        }});
        const data=await res.json();
        if(data.error){{
          document.getElementById('draw-conf')
            .textContent='Error: '+data.error;
          return;
        }}
        document.getElementById('draw-pred')
          .textContent=data.prediction;
        const conf=data.confidence;
        const color=conf>=90?'#4CAF50':
                    conf>=60?'#FF9800':'#f44336';
        const msg=conf>=90?'🎯 Very Confident!':
                  conf>=60?'🤔 Fairly Sure':
                           '😕 Try Again';
        document.getElementById('draw-conf')
          .innerHTML=
            `<span style="color:${{color}}">
              ${{conf.toFixed(1)}}% — ${{msg}}
             </span>`;
        const sp=data.spike_counts;
        const mx=Math.max(...sp,1);
        document.getElementById('draw-sbar')
          .innerHTML=sp.map((s,i)=>`
            <div class="sc">
              <div class="sf" style="
                height:${{Math.round(s/mx*65)+2}}px;
                background:${{i==data.prediction?
                  '#E91E63':
                  'rgba(255,255,255,0.1)'}}>
              </div>
              <div class="sl">${{i}}</div>
            </div>`).join('');
        document.getElementById('draw-extra')
          .innerHTML=
            `<span class="tag">
              Spikes: ${{data.total_spikes}}
             </span>`;
      }}catch(e){{
        document.getElementById('draw-conf')
          .textContent='Error: '+e.message;
      }}
    }}
    </script>
    </body></html>"""
    return html


@app.route('/predict_drawing', methods=['POST'])
def predict_drawing():
    try:
        data      = request.get_json()
        image_b64 = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_b64)
        img_pil   = Image.open(
            io.BytesIO(img_bytes)).convert('L')
        img_pil   = img_pil.resize((28, 28))
        img_arr   = np.array(img_pil).astype(
            np.float32) / 255.0
        result = predict_image(model, img_arr)
        state['n_predictions'] += 1
        pred = result['prediction']
        conf = float(
            result['confidence'][pred] * 100)
        state['history'].insert(0, {
            'prediction': int(pred),
            'confidence': conf,
            'source':     'draw',
        })
        state['history'] = state['history'][:10]
        return jsonify({
            'prediction':   int(pred),
            'confidence':   conf,
            'spike_counts': [float(x) for x in
                             result['spike_counts']],
            'total_spikes': int(
                result['spike_counts'].sum()),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/test_sample')
def test_sample():
    import random
    _, test_loader = load_mnist(batch_size=512)
    imgs, labels   = next(iter(test_loader))
    idx    = random.randint(0, len(imgs)-1)
    img    = imgs[idx].squeeze().numpy()
    label  = labels[idx].item()
    result = predict_image(model, img)
    state['n_predictions'] += 1
    pred    = result['prediction']
    conf    = result['confidence'][pred] * 100
    spikes  = result['spike_counts']
    max_s   = max(spikes.max(), 1)
    correct = pred == label

    state['history'].insert(0, {
        'prediction': int(pred),
        'true':       int(label),
        'confidence': float(conf),
        'correct':    bool(correct),
        'source':     'test',
    })
    state['history'] = state['history'][:10]

    img_b64    = arr_to_b64(img)
    spike_bars = ''.join([
        f'<div class="sc"><div class="sf" '
        f'style="height:{int(s/max_s*65)+2}px;'
        f'background:{"#E91E63" if i==pred else "rgba(255,255,255,0.1)"};">'
        f'</div><div class="sl">{i}</div></div>'
        for i, s in enumerate(spikes)
    ])

    html = f"""<!DOCTYPE html><html><head>
    <title>Test</title>{STYLE}</head>
    <body>{NAV}<div class="wrap">
    <div class="card" style="border-color:
         {'rgba(76,175,80,0.4)' if correct
          else 'rgba(244,67,54,0.4)'}">
      <h2>🎲 RANDOM MNIST TEST</h2>
      <div class="g2">
        <div>
          <img src="data:image/png;base64,{img_b64}"
               style="width:140px;height:140px;
                      border-radius:8px">
          <div style="margin-top:16px">
            <div class="mv"
                 style="font-size:4rem;
                 color:{'#4CAF50' if correct
                        else '#f44336'}">
              {pred}
            </div>
            <div class="ml">
              PREDICTED | TRUE: {label}
            </div>
            <div style="margin-top:10px;
                        display:flex;gap:8px">
              <span class="tag
                    {'tag-g' if correct else ''}"
                    style="{'border-color:#f44336;'
                            'color:#f44336'
                            if not correct else ''}">
                {'✅ CORRECT'
                 if correct else '❌ WRONG'}
              </span>
              <span class="tag tag-g">
                {conf:.1f}%
              </span>
            </div>
          </div>
        </div>
        <div>
          <p style="color:#90CAF9;
                    margin-bottom:8px">
            SPIKE COUNTS
          </p>
          <div class="sbar">{spike_bars}</div>
        </div>
      </div>
    </div>
    <div style="text-align:center;display:flex;
                gap:12px;justify-content:center">
      <a href="/test_sample" class="btn">
        🎲 Another</a>
      <a href="/draw" class="btn btn-b">
        ✏️ Draw</a>
      <a href="/spikes" class="btn btn-g">
        ⚡ Spikes</a>
    </div>
    </div></body></html>"""
    return html


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result_html = ''
    if (request.method == 'POST' and
            'file' in request.files):
        f = request.files['file']
        try:
            img_pil = Image.open(
                f.stream).convert('L').resize(
                (28, 28))
            img_arr = np.array(img_pil).astype(
                np.float32) / 255.0
            result  = predict_image(model, img_arr)
            state['n_predictions'] += 1
            pred   = result['prediction']
            conf   = result['confidence'][pred]*100
            spikes = result['spike_counts']
            max_s  = max(spikes.max(), 1)
            img_b64 = arr_to_b64(img_arr)
            spike_bars = ''.join([
                f'<div class="sc"><div class="sf" '
                f'style="height:{int(s/max_s*65)+2}px;'
                f'background:{"#E91E63" if i==pred else "rgba(255,255,255,0.1)"};">'
                f'</div><div class="sl">{i}</div></div>'
                for i, s in enumerate(spikes)
            ])
            result_html = f"""
            <div class="card" style="border-color:
                 rgba(233,30,99,0.4)">
              <h2>🎯 RESULT</h2>
              <div class="g2">
                <div>
                  <img src="data:image/png;
                       base64,{img_b64}"
                       style="width:140px;
                              height:140px;
                              border-radius:8px">
                  <div class="mv"
                       style="font-size:4rem;
                              margin-top:14px">
                    {pred}
                  </div>
                  <div class="ml">PREDICTED</div>
                  <span class="tag tag-g"
                        style="margin-top:10px;
                               display:inline-block">
                    {conf:.1f}% confidence
                  </span>
                </div>
                <div>
                  <p style="color:#90CAF9;
                            margin-bottom:8px">
                    SPIKE COUNTS
                  </p>
                  <div class="sbar">
                    {spike_bars}
                  </div>
                </div>
              </div>
            </div>"""
        except Exception as e:
            result_html = (
                f'<div class="card">'
                f'<p style="color:#f44336">'
                f'Error: {e}</p></div>')

    html = f"""<!DOCTYPE html><html><head>
    <title>Upload</title>{STYLE}</head>
    <body>{NAV}<div class="wrap">
    <div class="card">
      <h2>📤 UPLOAD IMAGE</h2>
      <form method="post"
            enctype="multipart/form-data">
        <div class="upload"
             onclick="document.getElementById(
               'fi').click()">
          <div style="font-size:2.5rem">📁</div>
          <p style="margin-top:10px">
            Click to select image
          </p>
          <p style="color:#555;font-size:.78rem">
            .png .jpg .bmp
          </p>
        </div>
        <input type="file" id="fi" name="file"
               accept="image/*"
               style="display:none"
               onchange="this.form.submit()">
      </form>
      <p style="color:#FF9800;font-size:.82rem;
                margin-top:12px">
        💡 Best: white digit on black background.
        Or use
        <a href="/draw" style="color:#E91E63">
          Draw feature!
        </a>
      </p>
    </div>
    {result_html}
    </div></body></html>"""
    return html


@app.route('/webcam')
def webcam_page():
    html = f"""<!DOCTYPE html><html><head>
    <title>Webcam</title>{STYLE}
    <style>
      #feed{{width:100%;border-radius:12px}}
      #pred{{font-size:5rem;color:#E91E63;
             font-weight:900;text-align:center;
             font-family:'Orbitron',sans-serif}}
    </style>
    </head><body>{NAV}<div class="wrap">
    <div class="g2">
      <div class="card">
        <h2>📷 LIVE WEBCAM</h2>
        <img id="feed" src="/video_feed">
        <div style="margin-top:14px;display:flex;
                    gap:10px;flex-wrap:wrap">
          <button class="btn"
                  onclick="capture()">
            📸 Capture
          </button>
          <button class="btn btn-b"
                  onclick="startAuto()">
            ▶ Auto
          </button>
          <button class="btn"
            style="background:rgba(255,255,255,0.1)"
            onclick="stopAuto()">
            ⏹ Stop
          </button>
        </div>
        <p style="color:#00BCD4;font-size:.82rem;
                  margin-top:10px">
          💡 Write digit LARGE with thick marker.
          Hold FLAT in yellow box!
        </p>
      </div>
      <div class="card">
        <h2>🎯 LIVE PREDICTION</h2>
        <div id="pred">—</div>
        <div id="conf"
             style="color:#90CAF9;
                    text-align:center;
                    margin:10px 0">
          Show digit to webcam...
        </div>
        <div id="sbar" class="sbar"></div>
        <div id="info"
             style="margin-top:12px;
                    text-align:center">
        </div>
      </div>
    </div>
    <script>
      let timer=null;
      async function capture(){{
        const r=await fetch('/capture',
          {{method:'POST'}});
        const d=await r.json();
        if(d.error) return;
        document.getElementById('pred')
          .textContent=d.prediction;
        const c=d.confidence;
        const color=c>=90?'#4CAF50':
                    c>=60?'#FF9800':'#f44336';
        document.getElementById('conf')
          .innerHTML=
            `<span style="color:${{color}}">
              ${{c.toFixed(1)}}%
             </span>`;
        const sp=d.spike_counts;
        const mx=Math.max(...sp,1);
        document.getElementById('sbar')
          .innerHTML=sp.map((s,i)=>`
            <div class="sc">
              <div class="sf" style="
                height:${{Math.round(s/mx*65)+2}}px;
                background:${{i==d.prediction?
                  '#E91E63':
                  'rgba(255,255,255,0.1)'}}>
              </div>
              <div class="sl">${{i}}</div>
            </div>`).join('');
      }}
      function startAuto(){{
        if(timer) return;
        timer=setInterval(capture,2000);
      }}
      function stopAuto(){{
        clearInterval(timer);timer=null;
      }}
    </script>
    </div></body></html>"""
    return html


@app.route('/video_feed')
def video_feed():
    def gen():
        import cv2
        cap=cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        if not cap.isOpened():
            ph=np.zeros((480,640,3),dtype=np.uint8)
            cv2.putText(ph,'No webcam',
              (200,240),
              cv2.FONT_HERSHEY_SIMPLEX,
              1,(255,255,255),2)
            _,j=cv2.imencode('.jpg',ph)
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg'
                   b'\r\n\r\n'
                   +j.tobytes()+b'\r\n')
            return
        try:
            while True:
                ret,frame=cap.read()
                if not ret: break
                h,w=frame.shape[:2]
                s=min(h,w)//3
                cx,cy=w//2,h//2
                cv2.rectangle(frame,
                  (cx-s//2,cy-s//2),
                  (cx+s//2,cy+s//2),
                  (0,255,255),2)
                cv2.putText(frame,
                  'Show digit here',
                  (cx-s//2,cy-s//2-10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.55,(0,255,255),1)
                _,j=cv2.imencode('.jpg',frame)
                yield (b'--frame\r\n'
                       b'Content-Type:'
                       b'image/jpeg\r\n\r\n'
                       +j.tobytes()+b'\r\n')
        finally:
            cap.release()
    return Response(gen(),
      mimetype='multipart/x-mixed-replace;'
               ' boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    try:
        import cv2
        from webcam_utils import preprocess_for_snn
        cap=cv2.VideoCapture(0)
        frames=[]
        if cap.isOpened():
            for _ in range(5):
                ret,frame=cap.read()
                if ret:
                    h,w=frame.shape[:2]
                    s=min(h,w)//3
                    cx,cy=w//2,h//2
                    roi=(cx-s//2,cy-s//2,s,s)
                    img=preprocess_for_snn(
                        frame,roi)
                    frames.append(img)
            cap.release()
        if frames:
            img=np.mean(frames,axis=0)
        else:
            _,tl=load_mnist(1)
            imgs,_=next(iter(tl))
            img=imgs[0].squeeze().numpy()
        result=predict_image(model,img)
        state['n_predictions']+=1
        pred=result['prediction']
        state['history'].insert(0,{
            'prediction':int(pred),
            'confidence':float(
                result['confidence'][pred]*100),
            'source':'webcam',
        })
        state['history']=state['history'][:10]
        return jsonify({
            'prediction':int(pred),
            'confidence':float(
                result['confidence'][pred]*100),
            'spike_counts':[float(x) for x in
                result['spike_counts']],
            'total_spikes':int(
                result['spike_counts'].sum()),
        })
    except Exception as e:
        return jsonify({'error':str(e)})


@app.route('/spikes')
def spikes_page():
    try:
        import random
        _,test_loader=load_mnist(batch_size=512)
        imgs,labels=next(iter(test_loader))
        idx=random.randint(0,len(imgs)-1)
        img=imgs[idx].squeeze().numpy()
        label=labels[idx].item()
        result=predict_image(model,img)
        pred=result['prediction']
        mem=result['mem_trace']

        fig,ax=plt.subplots(
            figsize=(10,4),facecolor='#0a0a18')
        ax.set_facecolor('#0a0a18')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        cmap=plt.cm.tab10
        for i in range(10):
            lw=2.5 if i==pred else 0.8
            al=1.0 if i==pred else 0.3
            ax.plot(
                mem[:,i] if mem.ndim==2 else mem,
                lw=lw,alpha=al,color=cmap(i),
                label=f'Digit {i}')
        ax.set_xlabel('Timestep',color='white')
        ax.set_ylabel('Membrane Potential',
                      color='white')
        ax.set_title(
            f'Membrane Voltages — Predicted:{pred}',
            color='white')
        ax.legend(loc='upper right',fontsize=7,
                  ncol=5,facecolor='#0a0a18',
                  labelcolor='white',
                  framealpha=0.5)
        mem_b64=fig_to_b64(fig)
        img_b64=arr_to_b64(img)
        spikes=result['spike_counts']
        max_s=max(spikes.max(),1)
        spike_bars=''.join([
            f'<div class="sc"><div class="sf" '
            f'style="height:{int(s/max_s*65)+2}px;'
            f'background:{"#E91E63" if i==pred else "rgba(255,255,255,0.1)"};">'
            f'</div><div class="sl">{i}</div></div>'
            for i,s in enumerate(spikes)
        ])
        html=f"""<!DOCTYPE html><html><head>
        <title>Spikes</title>{STYLE}</head>
        <body>{NAV}<div class="wrap">
        <div class="card">
          <h2>📈 MEMBRANE VOLTAGES</h2>
          <img src="data:image/png;base64,{mem_b64}"
               class="ri">
        </div>
        <div class="card">
          <h2>⚡ SPIKE COUNTS</h2>
          <div class="g2">
            <div>
              <img src="data:image/png;
                   base64,{img_b64}"
                   style="width:130px;
                          height:130px;
                          border-radius:8px">
              <div class="mv"
                   style="margin-top:12px">
                {pred}
              </div>
              <div class="ml">
                PREDICTED | TRUE:{label}
              </div>
            </div>
            <div>
              <div class="sbar">{spike_bars}</div>
              <div class="g3"
                   style="margin-top:14px">
                <div class="mbox">
                  <div class="mv"
                       style="font-size:1.5rem">
                    {int(spikes.sum())}
                  </div>
                  <div class="ml">SPIKES</div>
                </div>
                <div class="mbox">
                  <div class="mv"
                       style="font-size:1.5rem">
                    {result['confidence'][pred]*100:.0f}%
                  </div>
                  <div class="ml">CONFIDENCE</div>
                </div>
                <div class="mbox">
                  <div class="mv"
                       style="font-size:1.5rem">
                    25
                  </div>
                  <div class="ml">TIMESTEPS</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div style="text-align:center;
                    display:flex;gap:12px;
                    justify-content:center">
          <a href="/draw" class="btn">✏️ Draw</a>
          <a href="/test_sample" class="btn btn-b">
            🎲 Test</a>
          <a href="/metrics" class="btn btn-g">
            📊 Metrics</a>
        </div>
        </div></body></html>"""
        return html
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/metrics')
def metrics_page():
    m=run_metrics(state['accuracy'])
    snn=m['snn']
    cnn=m['cnn']
    ratio=m['ratio']
    img_b64=''
    if os.path.exists(
            'results/metrics_comparison.png'):
        with open(
                'results/metrics_comparison.png',
                'rb') as f:
            img_b64=base64.b64encode(
                f.read()).decode()
    tc_b64=''
    if os.path.exists(
            'results/training_curve.png'):
        with open('results/training_curve.png',
                  'rb') as f:
            tc_b64=base64.b64encode(
                f.read()).decode()
    html=f"""<!DOCTYPE html><html><head>
    <title>Metrics</title>{STYLE}</head>
    <body>{NAV}<div class="wrap">
    <div class="g3" style="margin-bottom:20px">
      <div class="mbox">
        <div class="mv">{ratio:.0f}x</div>
        <div class="ml">ENERGY VS CNN</div>
      </div>
      <div class="mbox">
        <div class="mv">
          {snn['total_nJ']:.4f}
        </div>
        <div class="ml">SNN ENERGY (nJ)</div>
      </div>
      <div class="mbox">
        <div class="mv">
          {state['accuracy']:.1f}%
        </div>
        <div class="ml">ACCURACY</div>
      </div>
    </div>
    <div class="card">
      <h2>📊 SNN VS CNN</h2>
      <table>
        <tr><th>Metric</th>
            <th>SNN</th><th>CNN</th></tr>
        <tr><td>Accuracy</td>
            <td><span class="tag tag-g">
              {state['accuracy']:.1f}%
            </span></td>
            <td><span class="tag">
              99.2%
            </span></td></tr>
        <tr><td>Energy</td>
            <td><span class="tag tag-g">
              {snn['total_nJ']:.4f} nJ
            </span></td>
            <td><span class="tag">
              {cnn['total_nJ']:.4f} nJ
            </span></td></tr>
        <tr><td>Savings</td>
            <td><span class="tag tag-g">
              {ratio:.0f}x better
            </span></td>
            <td><span class="tag">
              baseline
            </span></td></tr>
        <tr><td>Computation</td>
            <td><span class="tag">
              Spike-based
            </span></td>
            <td><span class="tag">
              Dense MACs
            </span></td></tr>
        <tr><td>Sparsity</td>
            <td><span class="tag tag-g">
              ~70% silent
            </span></td>
            <td><span class="tag">
              0%
            </span></td></tr>
      </table>
    </div>
    {'<div class="card"><h2>📈 METRICS CHART</h2><img src="data:image/png;base64,'+img_b64+'" class="ri"></div>' if img_b64 else ''}
    {'<div class="card"><h2>📈 TRAINING CURVE</h2><img src="data:image/png;base64,'+tc_b64+'" class="ri"></div>' if tc_b64 else ''}
    </div></body></html>"""
    return html


@app.route('/history')
def history_page():
    hist=state['history']
    rows=''
    for h in hist:
        pred=h.get('prediction','?')
        conf=h.get('confidence',0)
        source=h.get('source','unknown')
        correct=h.get('correct',None)
        color=('#4CAF50' if conf>=90 else
               '#FF9800' if conf>=60
               else '#f44336')
        rows+=f"""
        <div style="display:flex;
                    align-items:center;
                    gap:12px;padding:8px 12px;
                    border-radius:8px;
                    background:rgba(255,255,255,0.03);
                    margin-bottom:6px;
                    border:1px solid
                    rgba(255,255,255,0.05)">
          <div style="font-size:1.5rem;
                      font-family:'Orbitron',
                      sans-serif;
                      color:#E91E63;width:40px;
                      text-align:center">
            {pred}
          </div>
          <div style="flex:1">
            <div style="color:white;
                        font-size:.9rem">
              Predicted: <b>{pred}</b>
            </div>
            <div style="color:{color};
                        font-size:.82rem">
              {conf:.1f}% | {source}
            </div>
          </div>
          {f'<span style="color:{"#4CAF50" if correct else "#f44336"};font-weight:bold">{"✅" if correct else "❌"}</span>' if correct is not None else ''}
        </div>"""
    if not hist:
        rows=('<p style="color:#555;'
              'text-align:center;padding:20px">'
              'No predictions yet!</p>')
    html=f"""<!DOCTYPE html><html><head>
    <title>History</title>{STYLE}</head>
    <body>{NAV}<div class="wrap">
    <div class="card">
      <h2>📋 PREDICTION HISTORY</h2>
      <p style="color:#90CAF9;margin-bottom:14px;
                font-size:.82rem">
        Last {len(hist)} |
        Total: {state['n_predictions']}
      </p>
      {rows}
    </div>
    <div style="text-align:center;display:flex;
                gap:12px;justify-content:center">
      <a href="/draw" class="btn">✏️ Draw</a>
      <a href="/test_sample" class="btn btn-b">
        🎲 Test</a>
      <a href="/webcam" class="btn btn-g">
        📷 Webcam</a>
    </div>
    </div></body></html>"""
    return html


if __name__ == '__main__':
    init_model()
    print("\n"+"="*50)
    print("  🌐 Flask → http://localhost:5000")
    print("="*50)
    print("  /        → Home")
    print("  /draw    → ✏️  Draw")
    print("  /upload  → 📤 Upload")
    print("  /webcam  → 📷 Webcam")
    print("  /spikes  → ⚡ Spikes")
    print("  /metrics → 📊 Metrics")
    print("  /history → 📋 History")
    print("="*50+"\n")
    app.run(debug=False, host='0.0.0.0',
            port=5000, threaded=True)
