'use strict';
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const escapeHTML = (v) => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => Number(n).toFixed(3);
const label = name => ({'AKO-High-only':'AKO-High','AKO-Low-only':'AKO-Low','Random':'AKO-Random'}[name] || name);
const menu = $('.menu-button');
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded',String(open)); menu.setAttribute('aria-label',open?'Close navigation':'Open navigation');
  $('#nav-links').classList.toggle('open',open);
});
$$('#nav-links a').forEach(a => a.addEventListener('click',()=>{
  menu.setAttribute('aria-expanded','false'); menu.setAttribute('aria-label','Open navigation'); $('#nav-links').classList.remove('open');
}));
document.addEventListener('keydown', e => {if(e.key==='Escape'){menu.setAttribute('aria-expanded','false');$('#nav-links').classList.remove('open');}});
const navObserver = new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){$$('#nav-links a').forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id));}});},{rootMargin:'-15% 0px -65% 0px'});
$$('main>section[id]').forEach(section=>navObserver.observe(section));
const actions={
 stay:{title:'Refine within an existing representation',description:'Edit a selected source at its current level. Stay can also resume a retained Low branch; it does not require continuing from the most recent candidate.',nodes:'<span class="node high">H₀</span><span class="node-arrow">→</span><span class="node high">H₁</span><span class="node-arrow">→</span><span class="node new">H₂</span>'},
 lower:{title:'Export, validate, and continue at Low',description:'Export a supported native representation from the selected High candidate. Lower requires applicable cross-level diagnostic evidence. Retain the High source for possible later exploration.',nodes:'<span class="node high">H₁</span><span class="node-label">export →</span><span class="node low">L₀</span><span class="node-arrow">→</span><span class="node low new">L₁</span>'},
 rollback:{title:'Restore a retained High source',description:'Return to a retained higher-level candidate to reopen structural choices. The existing Low branch remains available. Rollback restores source; it does not reverse-compile edited native code.',nodes:'<span class="node low">L₁</span><span class="node-label">restore →</span><span class="node high">H₁</span><span class="node-arrow">→</span><span class="node new">H₂</span>'}
};
$$('[data-action]').forEach(button=>button.addEventListener('click',()=>{
 $$('[data-action]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 const action=actions[button.dataset.action];$('#action-title').textContent=action.title;$('#action-description').textContent=action.description;$('#action-nodes').innerHTML=action.nodes;
}));
$('#copy-citation').addEventListener('click',async()=>{
 const value=$('#bibtex').textContent;
 try{await navigator.clipboard.writeText(value);$('#copy-status').textContent='Copied';}
 catch{const selection=window.getSelection();const range=document.createRange();range.selectNodeContents($('#bibtex'));selection.removeAllRanges();selection.addRange(range);$('#copy-status').textContent='Select and copy the highlighted text';}
});
const methods=['AKO-High-only','AKO-Low-only','AKO-Fixed','AKO-Unrestricted','Random','KDA','AKO+TileCas','KDA+TileCas'];
let data, page=0;
const pageSize=10;
function renderPerformance(){
 const configuration=$('#configuration').value;
 const rows=methods.map(method=>data.summary.find(r=>r.configuration===configuration&&r.method===method));
 const maximum=Math.ceil(Math.max(...rows.map(r=>Number(r.mean_speedup))));
 $('#performance-chart').innerHTML=rows.map(r=>`<div class="bar-row ${r.method==='AKO+TileCas'?'featured':r.method==='KDA+TileCas'?'secondary':''}"><span class="bar-label">${label(r.method)}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${Number(r.mean_speedup)/maximum*100}%"></div></div><span class="bar-value">${r.mean_display}×</span></div>`).join('');
 $('#results-table').innerHTML=rows.map(r=>`<tr class="${r.method.includes('+TileCas')?'highlight':''}"><th scope="row">${label(r.method)}</th><td>${r.mean_display} ± ${fmt(r.sd_of_five_aggregate_repeat_means)}</td><td>${fmt(r.median_of_five_repeat_task_means)}</td><td>[${fmt(r.bootstrap_lower)}, ${fmt(r.bootstrap_upper)}]</td><td>${r.majority_success} / ${r.task_configuration_pairs}</td></tr>`).join('');
 $('#result-table-caption').textContent=configuration+' · SD across five aggregate repeat means; CI from paired workload bootstrap';
 $('#performance-caption').textContent=configuration==='Overall'?'Overall weights the four accelerator–model configurations equally. Failed-run endpoints contribute zero. The chart shows mean speedup; SD and bootstrap intervals are in the numerical table.':configuration+' · 50 workloads, five agent runs per workload. Failed-run endpoints contribute zero. The chart shows mean speedup; SD and bootstrap intervals are in the numerical table.';
}
const colors={'AKO-High-only':'#779abd','AKO-Low-only':'#d8a567','AKO-Fixed':'#6b9e92','AKO-Unrestricted':'#a897b7','Random':'#ba8092','AKO+TileCas':'#5746b2'};
function renderProgress(kind='time'){
 const W=970,H=295,L=49,R=25,T=20,B=43,iw=W-L-R,ih=H-T-B,xmax=kind==='time'?120:64,ymax=5.2;
 const x=n=>L+Number(n)/xmax*iw,y=n=>T+ih-Number(n)/ymax*ih;
 const ticks=kind==='time'?[0,15,30,45,60,75,90,105,120]:[0,8,16,24,32,40,48,56,64];
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="progress-title progress-desc"><title id="progress-title">Search progress by ${kind==='time'?'elapsed minutes':'candidate evaluations'}</title><desc id="progress-desc">Six post-step curves from Figure 6. TileCas reaches 4.836 times the reference speed at the full budget. Downloadable CSV provides all reporting coordinates.</desc>`;
 for(let n=0;n<=5;n++){svg+=`<line x1="${L}" y1="${y(n)}" x2="${W-R}" y2="${y(n)}" class="chart-grid"/><text x="${L-13}" y="${y(n)+3}" text-anchor="end" class="chart-axis">${n}×</text>`;}
 ticks.forEach(n=>{svg+=`<text x="${x(n)}" y="${H-B+21}" text-anchor="middle" class="chart-axis">${n}</text>`;});
 for(const [method,color] of Object.entries(colors)){
  const rows=data.budget.filter(r=>r.method===method&&r.budget_type===kind).sort((a,b)=>Number(a.budget)-Number(b.budget));
  let path='';rows.forEach((r,i)=>{path+=i?` H${x(r.budget).toFixed(2)} V${y(r.mean_best_so_far_speedup).toFixed(2)}`:`M${x(r.budget).toFixed(2)},${y(r.mean_best_so_far_speedup).toFixed(2)}`;});
  svg+=`<path d="${path}" stroke="${color}" class="chart-path ${method==='AKO+TileCas'?'featured':''}" ${method==='AKO-Unrestricted'?'stroke-dasharray="5 4"':''}><title>${label(method)}: ${rows.at(-1).mean_best_so_far_speedup}× at ${xmax} ${kind==='time'?'minutes':'evaluations'}</title></path>`;
 }
 svg+=`<text x="${L+iw/2}" y="${H-3}" text-anchor="middle" class="chart-axis">${kind==='time'?'Elapsed time (minutes)':'Candidate evaluations'}</text></svg>`;
 $('#progress-chart').innerHTML=svg;
 $('#chart-legend').innerHTML=Object.entries(colors).map(([method,color])=>`<span class="legend-item"><i style="--series-color:${color}" aria-hidden="true"></i>${label(method).replace('AKO-','').replace('AKO+','')}</span>`).join('');
}
function renderWorkloads(){
 const query=$('#workload-search').value.toLowerCase().trim();
 const filtered=data.workloads.filter(r=>r.method===$('#workload-method').value&&r.configuration===$('#workload-config').value&&r.workload.toLowerCase().includes(query)).sort((a,b)=>a.workload.localeCompare(b.workload,undefined,{numeric:true}));
 const total=filtered.length;page=Math.max(0,Math.min(page,Math.ceil(total/pageSize)-1));
 $('#workload-table').innerHTML=filtered.slice(page*pageSize,(page+1)*pageSize).map(r=>`<tr><th scope="row">${escapeHTML(r.workload)}</th><td>${fmt(r.mean)}×</td><td>${fmt(r.final_mean)}×</td><td>${r.correct} / ${r.repeats}</td><td>${r.correct>=3?'Yes':'No'}</td></tr>`).join('')||'<tr><td colspan="5" class="empty-row">No workloads match this search.</td></tr>';
 $('#workload-count').textContent=total?`${page*pageSize+1}–${Math.min((page+1)*pageSize,total)} of ${total} workloads`:'0 workloads';
 $('#previous-page').disabled=page===0;$('#next-page').disabled=(page+1)*pageSize>=total;
}
async function initResults(){
 try{
  const response=await fetch('data/results.json');if(!response.ok)throw new Error('Data request failed');data=await response.json();
  renderPerformance();renderProgress();
  const allMethods=[...new Set(data.workloads.map(r=>r.method))];
  $('#workload-method').innerHTML=allMethods.map(m=>`<option value="${escapeHTML(m)}">${label(m)}</option>`).join('');$('#workload-method').value='AKO+TileCas';renderWorkloads();
  $('#configuration').addEventListener('change',renderPerformance);
  $$('[data-budget]').forEach(button=>button.addEventListener('click',()=>{$$('[data-budget]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderProgress(button.dataset.budget);}));
  ['#workload-method','#workload-config','#workload-search'].forEach(s=>$(s).addEventListener(s.includes('search')?'input':'change',()=>{page=0;renderWorkloads();}));
  $('#previous-page').addEventListener('click',()=>{page--;renderWorkloads();});$('#next-page').addEventListener('click',()=>{page++;renderWorkloads();});
 }catch(error){
  $('#performance-chart').innerHTML='<p class="error-message">Interactive data could not load. <a href="downloads/results.csv">Download the results CSV</a> or open the paper to view all results.</p>';
  $('#progress-chart').innerHTML='<p class="error-message">The search-progress coordinates are available in the CSV download below.</p>';
  $('#workload-table').innerHTML='<tr><td colspan="5">Data unavailable. <a href="downloads/seed-results.csv">Download seed-level results.</a></td></tr>';
 }
}
initResults();
