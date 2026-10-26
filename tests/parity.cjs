const assert=require('node:assert/strict'),fs=require('node:fs');
const {predict,corruption}=require('../app.js');
const m=JSON.parse(fs.readFileSync('reports/model.json')),r=JSON.parse(fs.readFileSync('reports/metrics.json'));
r.sample_predictions.forEach(s=>{const p=predict(s.pixels,m);assert.ok(Math.abs(Math.max(...p)-s.confidence)<1e-10);assert.equal(p.indexOf(Math.max(...p)),s.prediction);assert.ok(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-12);});
assert.throws(()=>predict([NaN],m));assert.throws(()=>predict(Array(64).fill(2),m));
assert.deepEqual(corruption(Array(64).fill(1),'shift').filter((v,i)=>i%8===0),Array(8).fill(0));
console.log('Browser/Python neural inference parity passed for 40 held-out images.');
