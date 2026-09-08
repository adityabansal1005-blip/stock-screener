/* Model output is always textContent: never interpreted as HTML or script. */
(() => {
  let generation = 0;
  const originalShow = window.showDetail;
  const originalClose = window.closeDetail;
  async function api(path, options) {
    const response = await fetch('/api/trading-research' + path, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Research request failed.');
    return body;
  }
  function element(tag, text, parent) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  window.closeDetail = function () { generation++; originalClose(); };
  window.showDetail = function (symbol) {
    originalShow(symbol);
    const token = ++generation;
    const old = document.getElementById('trading-research-panel');
    if (old) old.remove();
    const panel = element('section', '', document.getElementById('detail-content'));
    panel.id = 'trading-research-panel';
    panel.className = 'detail-section';
    panel.style.cssText = 'margin-top:24px;border-top:1px solid var(--border);padding-top:16px';
    element('h4', 'TradingAgents · Gemini research', panel);
    element('p', 'On-demand research using public market data. Uses your Google API quota; takes a few minutes. Does not change scores or place trades.', panel).style.fontSize = '12px';
    const button = element('button', 'Research this stock', panel);
    button.style.cssText = 'padding:9px 14px;cursor:pointer;border-radius:6px';
    const message = element('p', 'Checking configuration…', panel);
    message.setAttribute('role', 'status');
    message.style.fontSize = '12px';
    const output = element('div', '', panel);
    const history = element('div', '', panel);
    const valid = () => token === generation;
    function render(job) {
      output.replaceChildren();
      message.textContent = `${job.ticker} · ${job.status} · ${job.created_at} · ${job.model}`;
      if (job.error) element('p', job.error, output);
      if (job.status !== 'completed') return;
      element('p', 'AI opinion: ' + (typeof job.decision === 'string' ? job.decision : JSON.stringify(job.decision)), output);
      element('p', job.limitations, output).style.cssText = 'font-size:12px;color:var(--muted)';
      if (job.price_source) element('p', `Prices: ${job.price_source.source} · latest completed bar ${job.price_source.last_bar} · close ₹${job.price_source.daily_close}`, output);
      Object.entries(job.reports || {}).forEach(([name, report]) => {
        if (!report) return;
        const details = element('details', '', output);
        element('summary', name.replaceAll('_', ' '), details);
        const body = element('pre', typeof report === 'string' ? report : JSON.stringify(report, null, 2), details);
        body.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:12px;line-height:1.6';
      });
      const download = element('button', 'Download report', output);
      download.onclick = () => {
        const blob = new Blob([JSON.stringify(job, null, 2)], {type:'application/json'});
        const url = URL.createObjectURL(blob);
        const a = element('a'); a.href = url; a.download = `${job.ticker}-${job.analysis_date}-research.json`;
        a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      };
    }
    async function watch(id) {
      try {
        const job = await api('/jobs/' + id);
        if (!valid()) return;
        render(job);
        button.disabled = job.status === 'running';
        if (job.status === 'running') setTimeout(() => { if (valid()) watch(id); }, 3000);
        else loadHistory(false);
      } catch (error) {
        if (valid()) { message.textContent = error.message; button.disabled = false; }
      }
    }
    async function loadHistory(openLatest) {
      try {
        const jobs = await api('/jobs?symbol=' + encodeURIComponent(symbol));
        if (!valid()) return;
        history.replaceChildren();
        if (jobs.length) element('h4', 'Saved research', history);
        jobs.slice(0, 5).forEach(job => {
          const view = element('button', `${job.created_at.slice(0,16).replace('T',' ')} · ${job.status}`, history);
          view.style.cssText = 'margin:4px;padding:5px;cursor:pointer';
          view.onclick = () => watch(job.id);
        });
        if (openLatest && jobs.length) watch(jobs[0].id);
      } catch (error) { if (valid()) message.textContent = error.message; }
    }
    button.onclick = async () => {
      button.disabled = true; message.textContent = 'Starting research…';
      try {
        const job = await api('/start?symbol=' + encodeURIComponent(symbol), {
          method:'POST', headers:{'X-Research-Action':'start'}
        });
        if (valid()) watch(job.id);
      } catch (error) {
        if (valid()) { message.textContent = error.message; button.disabled = false; }
      }
    };
    api('/status').then(status => {
      if (!valid()) return;
      message.textContent = !status.installed ? 'TradingAgents environment needs installation.' :
        !status.key_configured ? 'Add GOOGLE_API_KEY to D:\\stock-screener-by-codex\\.env, then click Research this stock.' :
        `Ready · ${status.model}. Completed reports are reused for 6 hours.`;
      loadHistory(true);
    }).catch(error => { if (valid()) message.textContent = error.message; });
  };
})();
