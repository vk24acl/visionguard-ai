function logits(x,m){
  if(x.length!==64||x.some(v=>!Number.isFinite(v)||v<0||v>1))throw Error('Expected 64 pixels in [0,1]');
  let h=x.slice();
  m.weights.forEach((w,i)=>{h=m.biases[i].map((b,j)=>b+h.reduce((sum,v,k)=>sum+v*w[k][j],0));if(i<m.weights.length-1)h=h.map(v=>Math.max(0,v));});
  return h;
}
function predict(x,m){const z=logits(x,m).map(v=>v/m.temperature),max=Math.max(...z),ex=z.map(v=>Math.exp(v-max)),sum=ex.reduce((a,b)=>a+b,0);return ex.map(v=>v/sum);}
function corruption(x,kind){let a=x.slice();if(kind==='occlusion')for(let i=24;i<40;i++)a[i]=0;if(kind==='shift')a=a.map((v,i)=>i%8===0?0:x[i-1]);return a;}
if(typeof module!=='undefined')module.exports={logits,predict,corruption};
if(typeof document!=='undefined'){
 const r=window.REPORT,m=window.MODEL,pct=x=>(100*x).toFixed(1)+'%';
 document.getElementById('metrics').innerHTML=[['Clean test accuracy',pct(r.conditions.clean.accuracy)],['Accepted coverage',pct(r.conditions.clean.coverage)],['Accepted accuracy',pct(r.conditions.clean.selective_accuracy)],['Test images','270']].map(([a,b])=>`<div class="metric"><small>${a}</small><strong>${b}</strong></div>`).join('');
 document.getElementById('robustness').innerHTML='<table><thead><tr><th>Condition</th><th>Accuracy</th><th>Coverage</th><th>Accepted accuracy</th></tr></thead><tbody>'+Object.entries(r.conditions).map(([k,v])=>`<tr><td>${k}</td><td>${pct(v.accuracy)}</td><td>${pct(v.coverage)}</td><td>${v.selective_accuracy==null?'—':pct(v.selective_accuracy)}</td></tr>`).join('')+'</tbody></table>';
 document.getElementById('risk').innerHTML=r.risk_coverage.map(a=>`<div class="bar"><span>Confidence ≥ ${a.threshold.toFixed(2)}</span><i style="width:${a.coverage*180}px"></i><b>${pct(a.coverage)} covered; ${a.selective_accuracy==null?'—':pct(a.selective_accuracy)} correct</b></div>`).join('');
 const canvas=document.getElementById('digit'),ctx=canvas.getContext('2d');let pixels=r.sample_predictions[0].pixels.slice(),truth=r.sample_predictions[0].label;
 function redraw(){const a=corruption(pixels,document.getElementById('condition').value);for(let i=0;i<64;i++){const g=Math.round(a[i]*255);ctx.fillStyle=`rgb(${g},${g},${g})`;ctx.fillRect((i%8)*32,Math.floor(i/8)*32,32,32);}const p=predict(a,m),best=p.indexOf(Math.max(...p));document.getElementById('prediction').textContent=`Predicted ${best} · confidence ${pct(p[best])} · ${p[best]>=m.threshold?'accepted':'abstained'}${truth==null?' · custom drawing, unvalidated distribution':' · true label '+truth}`;document.getElementById('probabilities').innerHTML=p.map((v,i)=>`<div class="bar"><span>${i}</span><i style="width:${v*180}px"></i><b>${pct(v)}</b></div>`).join('');}
 const select=document.getElementById('sample');select.innerHTML=r.sample_predictions.map((a,i)=>`<option value="${i}">Test sample ${i+1} — true ${a.label}</option>`).join('');select.onchange=()=>{pixels=r.sample_predictions[+select.value].pixels.slice();truth=r.sample_predictions[+select.value].label;redraw();};
 document.getElementById('condition').onchange=redraw;document.getElementById('clear').onclick=()=>{pixels=Array(64).fill(0);truth=null;redraw();};
 let down=false;function draw(e){if(!down)return;const b=canvas.getBoundingClientRect(),x=Math.min(7,Math.max(0,Math.floor((e.clientX-b.left)/b.width*8))),y=Math.min(7,Math.max(0,Math.floor((e.clientY-b.top)/b.height*8)));pixels[y*8+x]=1;truth=null;redraw();}
 canvas.onpointerdown=e=>{down=true;canvas.setPointerCapture(e.pointerId);draw(e);};canvas.onpointermove=draw;canvas.onpointerup=()=>down=false;canvas.onpointercancel=()=>down=false;redraw();
}
