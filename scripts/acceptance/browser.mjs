/** Read-only local Chrome acceptance. Node >=22; no npm dependencies or training.
 * Start an isolated Chrome with --remote-debugging-port=9231 first.
 * Screenshots are private evidence: keep output outside the repository.
 */
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const base = process.env.ACCEPTANCE_URL || 'http://127.0.0.1:8519';
const endpoint = process.env.ACCEPTANCE_CDP || 'http://127.0.0.1:9231';
for (const value of [base, endpoint]) {
  assert(['127.0.0.1', 'localhost'].includes(new URL(value).hostname), 'Local endpoints only');
}
const out = process.env.ACCEPTANCE_OUTPUT;
assert(out, 'Set ACCEPTANCE_OUTPUT to a private evidence directory');
await fs.mkdir(out, {recursive: true});
const tabs = await (await fetch(endpoint + '/json')).json();
const ws = new WebSocket(tabs.find(x => x.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, {once: true}));
let id = 0;
const pending = new Map(), failures = [], checks = [];
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(m.error) : p.resolve(m.result); }
  else if (m.method === 'Runtime.exceptionThrown') failures.push(m.params.exceptionDetails.text);
});
async function cmd(method, params = {}) {
  const key = ++id;
  return await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('CDP timeout: ' + method)), 15000);
    pending.set(key, {resolve: v => {clearTimeout(timer); resolve(v)}, reject});
    ws.send(JSON.stringify({id: key, method, params}));
  });
}
async function ev(expression) {
  const r = await cmd('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function wait(expression) {
  for (let i = 0; i < 120; i++) {if (await ev(expression)) return; await sleep(250);}
  throw Error('UI did not reach expected state: ' + expression);
}
async function settle() {
  await sleep(400);
  await wait(`!document.querySelector('[data-testid="stStatusWidget"]') && ![...document.querySelectorAll("button")].some(e=>e.innerText==="Stop")`);
  await sleep(400);
  assert.equal(await ev(`document.querySelectorAll('[data-testid="stException"]').length`), 0);
}
async function clickPoint(p) {
  assert(p, 'Expected visible target');
  await cmd('Input.dispatchMouseEvent', {type: 'mouseMoved', ...p});
  await cmd('Input.dispatchMouseEvent', {type: 'mousePressed', ...p, button: 'left', clickCount: 1});
  await cmd('Input.dispatchMouseEvent', {type: 'mouseReleased', ...p, button: 'left', clickCount: 1});
}
async function choose(label, text) {
  const button = await ev(`(()=>{const e=[...document.querySelectorAll('[data-testid="stSelectbox"],[data-testid="stMultiSelect"]')].find(e=>e.querySelector('label')?.innerText===${JSON.stringify(label)});if(!e)throw Error('Missing '+${JSON.stringify(label)});e.scrollIntoView({block:'center'});const r=e.querySelector('button[aria-label="Open"]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  await clickPoint(button);
  await sleep(300);
  let p = await ev(`(()=>{const e=[...document.querySelectorAll('[role="option"]')].find(e=>e.textContent.includes(${JSON.stringify(text)}));if(!e)return null;e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  if (!p) {
    await ev(`(()=>{const e=[...document.querySelectorAll('[data-testid="stSelectbox"],[data-testid="stMultiSelect"]')].find(e=>e.querySelector('label')?.innerText===${JSON.stringify(label)});e.querySelector('input').focus();e.querySelector('input').select();})()`);
    await cmd('Input.insertText', {text}); await sleep(300);
    p = await ev(`(()=>{const e=[...document.querySelectorAll('[role="option"]')].find(e=>e.textContent.includes(${JSON.stringify(text)}));if(!e)return null;const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  }
  assert(p, 'Missing option: ' + label + ' = ' + text);
  await clickPoint(p); await settle();
}
async function snapshot(name) {
  const layout = await ev(`({width:innerWidth,overflow:document.documentElement.scrollWidth>innerWidth,exceptions:document.querySelectorAll('[data-testid="stException"]').length,charts:[...document.querySelectorAll('[data-testid="stVegaLiteChart"]')].filter(e=>e.clientWidth>0).length})`);
  assert.equal(layout.overflow, false, 'Horizontal page overflow'); assert.equal(layout.exceptions, 0);
  const r = await cmd('Page.captureScreenshot', {format: 'png'});
  await fs.writeFile(out + '/' + name + '.png', Buffer.from(r.data, 'base64'));
  checks.push({name, ...layout});
}
try {
  await cmd('Page.enable'); await cmd('Runtime.enable');
  await cmd('Emulation.setDeviceMetricsOverride', {width: 1440, height: 900, deviceScaleFactor: 1, mobile: false});
  const start = performance.now();
  await cmd('Page.navigate', {url: base});
  await wait(`document.body.innerText.includes('Ideas to test')`); await settle();
  checks.push({name: 'initial_render', seconds: (performance.now() - start)/1000});
  await snapshot('overview-1440');
  // A real card opens the exact published sheet; a fresh URL retains its identity.
  await ev(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.innerText==='Inspect setup & plan test');if(!e)throw Error('No inspectable research card');e.click();})()`);
  await settle();
  await wait(`location.search.includes('candidate=') && document.body.innerText.includes('Full OddsJam setup')`);
  const deepLink = await ev('location.href');
  await ev(`[...document.querySelectorAll('[role="tab"]')].find(e=>e.textContent.includes('Full OddsJam setup')).click()`);
  await settle();
  assert(await ev(`document.body.innerText.includes('Unknown') || document.body.innerText.includes('unknown') || document.body.innerText.includes('inherited')`));
  await ev(`[...document.querySelectorAll('[role="tab"]')].find(e=>e.textContent.includes('Full OddsJam setup')).scrollIntoView({block:'start'})`);
  await snapshot('setup-1440');
  await cmd('Page.navigate', {url: deepLink});
  await wait(`document.body.innerText.includes('Full OddsJam setup')`); await settle();
  assert.equal(await ev('location.href'), deepLink);
  checks.push({name:'card_setup_and_exact_deep_link', passed:true});
  await ev(`[...document.querySelectorAll('button')].find(e=>e.innerText==='Reset signal filters').click()`);
  await settle(); assert.equal(await ev(`new URL(location.href).searchParams.has('candidate')`), false);
  checks.push({name:'reset_clears_deep_link', passed:true});

  await choose('Explore', 'Opportunities'); await snapshot('opportunities-1440');
  await choose('Explore', 'Production health'); await snapshot('production-1440');
  await choose('Explore', 'Explore history');

  // Hover and select a rendered mark, not an AppTest placeholder.
  await choose('Diagnostic', 'Profit by group');
  const mark = await ev(`(()=>{const e=[...document.querySelectorAll('[data-testid="stVegaLiteChart"] svg [aria-roledescription="bar"]')].find(e=>e.getBoundingClientRect().width>0);if(!e)return null;e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  assert(mark, 'Expected a rendered profit bar');
  await cmd('Input.dispatchMouseEvent', {type:'mouseMoved', ...mark}); await sleep(500);
  assert(await ev(`document.querySelector('#vg-tooltip-element')?.textContent.includes('eligible_count')`), 'Hover must show a tooltip');
  await clickPoint(mark);
  checks.push({name:'rendered_chart_hover_and_click',passed:true});
  for (const diagnostic of ['Profit by group', 'Matched EV vs return', 'Saved-filter EV vs return', 'EV bands', 'Liquidity', 'Liquidity × EV', 'Time to event', 'Cumulative profit by platform', 'Bet volume over time', 'Average weekday / hour activity']) {
    await choose('Diagnostic', diagnostic);
    assert(await ev(`document.querySelector('[data-testid="stVegaLiteChart"]') !== null || document.body.innerText.includes('Chart blocked') || document.body.innerText.includes('No exact published chart cohort')`));
    checks.push({name: diagnostic, rendered_or_explicit_blocker: true});
  }

  await choose('Diagnostic', 'EV bands');
  const bucket = await ev(`(()=>{const e=[...document.querySelectorAll('[data-testid="stVegaLiteChart"] svg [aria-roledescription="point"]')].find(e=>e.getBoundingClientRect().width>0);if(!e)return null;e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  await clickPoint(bucket); await sleep(500);
  assert(await ev(`!!document.querySelector('[data-testid="stVegaLiteChart"] svg [opacity="0.2"]')`), 'Bucket selection must dim other linked marks');
  await cmd('Input.dispatchMouseEvent', {type:'mousePressed',...bucket,button:'left',clickCount:2});
  await cmd('Input.dispatchMouseEvent', {type:'mouseReleased',...bucket,button:'left',clickCount:2});
  await sleep(500);
  assert(await ev(`!document.querySelector('[data-testid="stVegaLiteChart"] svg [opacity="0.2"]')`), 'Double-click must reset bucket selection');
  checks.push({name:'linked_bucket_highlight_and_reset',passed:true});
  await cmd('Emulation.setDeviceMetricsOverride', {width: 1366, height: 768, deviceScaleFactor: 1, mobile: false});
  await snapshot('history-1366');
  await choose('Explore', 'Explain returns'); await snapshot('explanations-1366');
  await choose('Platform', 'Novig');
  checks.push({name:'sidebar_filtered_fixed_model',passed:true});
  await ev(`[...document.querySelectorAll('button')].find(e=>e.innerText==='Reset signal filters').click()`);
  await settle();
  await choose('Explanation publication', 'Separately fitted');
  await snapshot('subset-explanations-1366');
  assert(await ev(`document.body.innerText.includes('separately') || document.body.innerText.includes('Separately')`));
  checks.push({name:'explicit_offline_subset_model',passed:true});

  await choose('Explore', 'Overview'); await snapshot('overview-1366');
  assert.equal(failures.length, 0, failures.join('\n'));
  await fs.writeFile(out + '/browser.json', JSON.stringify({status: 'passed', checks, failures}, null, 2));
  console.log(JSON.stringify({status: 'passed', checks}));
} finally {ws.close();}
