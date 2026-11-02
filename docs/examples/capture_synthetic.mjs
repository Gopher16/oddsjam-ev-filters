/** Local synthetic screenshot capture. Node >=22, isolated Chrome CDP; no npm packages. */
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const root = await fs.realpath(process.env.WALKTHROUGH_ROOT || '');
assert.deepEqual(JSON.parse(await fs.readFile(path.join(root, 'SYNTHETIC_ONLY.json'))), {kind:'operator-demo',version:1});
const config = await fs.readFile(path.join(root, 'dashboard-monitored.yaml'), 'utf8');
assert(config.includes(root) && config.includes('lifecycle_report_publication:'), 'Use the completed synthetic workspace');
const base = process.env.TUTORIAL_URL || 'http://127.0.0.1:8541';
const cdp = process.env.TUTORIAL_CDP || 'http://127.0.0.1:9235';
for (const value of [base, cdp]) assert(['127.0.0.1','localhost'].includes(new URL(value).hostname));
const output = process.env.TUTORIAL_SCREENSHOTS;
assert(output, 'Set TUTORIAL_SCREENSHOTS to an explicit screenshot output directory');
await fs.mkdir(output, {recursive:true});
const tabs = await (await fetch(cdp+'/json')).json();
const ws = new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0; const pending=new Map();
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}});
async function cmd(method,params={}) {const n=++id;return await new Promise((resolve,reject)=>{
 const timer=setTimeout(()=>reject(Error('CDP timeout '+method)),15000);
 pending.set(n,{resolve:v=>{clearTimeout(timer);resolve(v)},reject});ws.send(JSON.stringify({id:n,method,params}));});}
async function ev(expression){const r=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function wait(expr){for(let i=0;i<120;i++){if(await ev(expr))return;await sleep(250)}throw Error('UI timeout '+expr)}
async function settle(){await sleep(500);await wait(`![...document.querySelectorAll('button')].some(e=>e.innerText==='Stop')`);await sleep(500);assert.equal(await ev(`document.querySelectorAll('[data-testid="stException"]').length`),0)}
async function point(p){assert(p,'Missing visible target');await cmd('Input.dispatchMouseEvent',{type:'mouseMoved',...p});await cmd('Input.dispatchMouseEvent',{type:'mousePressed',...p,button:'left',clickCount:1});await cmd('Input.dispatchMouseEvent',{type:'mouseReleased',...p,button:'left',clickCount:1})}
async function choose(label,text){
 const button=await ev(`(()=>{const e=[...document.querySelectorAll('[data-testid="stSelectbox"]')].find(e=>e.querySelector('label')?.innerText===${JSON.stringify(label)});if(!e)return null;e.scrollIntoView({block:'center'});const r=e.querySelector('button[aria-label="Open"]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
 await point(button);await sleep(300);
 const option=await ev(`(()=>{const e=[...document.querySelectorAll('[role="option"]')].find(e=>e.textContent.includes(${JSON.stringify(text)}));if(!e)return null;e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
 await point(option);await settle();
 await wait(`([...document.querySelectorAll('[data-testid="stSelectbox"]')].find(e=>e.querySelector('label')?.innerText===${JSON.stringify(label)})?.querySelector('input')?.value || '').includes(${JSON.stringify(text)})`);
}
async function scrollText(text){await ev(`(()=>{const e=[...document.querySelectorAll('h2,h3,h4,[data-testid="stMarkdownContainer"]')].find(e=>e.textContent.includes(${JSON.stringify(text)}));if(!e)throw Error('Missing section');e.scrollIntoView({block:'start'})})()`);await sleep(300)}
async function shot(name){
 await ev(`(()=>{let e=document.getElementById('synthetic-capture-label');if(!e){e=document.createElement('div');e.id='synthetic-capture-label';e.textContent='SYNTHETIC TUTORIAL · FICTIONAL DATA';e.style.cssText='position:fixed;bottom:8px;right:12px;z-index:999999;background:#fff4cb;color:#242424;padding:8px 12px;border:2px solid #b68400;font:600 13px sans-serif';document.body.appendChild(e)}})()`);
 assert.equal(await ev('document.documentElement.scrollWidth>innerWidth'),false);
 const r=await cmd('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(output,name+'.png'),Buffer.from(r.data,'base64'));
}
try {
 await cmd('Page.enable');await cmd('Runtime.enable');
 await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
 await cmd('Page.navigate',{url:base});await wait(`document.body.innerText.includes('Ideas to test')`);await settle();
 assert(await ev(`document.body.innerText.includes('Demo A')`),'Refusing capture without expected fictional platform');
 await choose('Saved filter','TEST - Synthetic parent');
 await scrollText('Next tests & production review');await shot('overview');
 await ev(`[...document.querySelectorAll('button')].find(e=>e.innerText==='Inspect setup & plan test').click()`);await settle();
 await wait(`document.body.innerText.includes('Full OddsJam setup')`);
 await ev(`[...document.querySelectorAll('[role="tab"]')].find(e=>e.textContent==='Full OddsJam setup').click()`);
 await ev(`[...document.querySelectorAll('[role="tab"]')].find(e=>e.textContent==='Full OddsJam setup').scrollIntoView({block:'start'})`);
 await shot('setup');
 await choose('Explore','Paper-test definitions');await scrollText('Parent, subgroup and complement economics');await shot('paper-study');
 await scrollText('Reviewed outcomes and configuration changes');await shot('lifecycle');
 console.log(JSON.stringify({synthetic_only:true,screenshots:['overview','setup','paper-study','lifecycle'],output}));
} finally {ws.close()}
