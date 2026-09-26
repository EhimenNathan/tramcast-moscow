// Tram load forecast dashboard (vanilla JS; Leaflet + Chart.js)
const $ = id => document.getElementById(id);
const fmt = v => Math.round(v).toLocaleString('ru-RU');
window._map = () => map;
let META = null, ROUTES = [], STOPS = {}, tsChart = null, profChart = null, map = null, layer = null, playing = null;

function toast(msg) { const t = $('toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', 6000); }

async function api(path) {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) { const e = body.error || {}; throw new Error((e.message || r.statusText) + (e.hint ? ' — ' + e.hint : '')); }
  return body;
}

function params(extra = {}) {
  const p = new URLSearchParams({
    route: $('route').value, date_from: $('dfrom').value, date_to: $('dto').value,
    hour_from: $('h1').value, hour_to: $('h2').value, agg: $('agg').value, horizon: $('horizon').value,
    k_weather: $('kw').value, k_event: $('ke').value, k_season: $('ks').value, ...extra
  });
  if (+$('pr').value >= 0) p.set('precip_mm', $('pr').value);
  if ($('stop').value) p.set('stop_id', $('stop').value);
  else if ($('secA').value && $('secB').value) { p.set('section_from', $('secA').value); p.set('section_to', $('secB').value); }
  return p;
}

function onHorizon() {
  const h = $('horizon').value;
  if (h === 'day') { $('dfrom').value = '2025-11-05'; $('dto').value = '2025-11-05'; $('agg').value = 'hour'; }
  if (h === 'month') { $('dfrom').value = '2025-11-01'; $('dto').value = '2025-11-30'; $('agg').value = 'day'; }
  if (h === 'year') { $('dfrom').value = '2026-01-01'; $('dto').value = '2026-12-31'; $('agg').value = 'month'; }
  refresh();
}

async function refresh() {
  ['kw', 'ke', 'ks'].forEach(k => $(k + '_v').textContent = (+$(k).value).toFixed(2));
  $('pr_v').textContent = +$('pr').value < 0 ? 'факт' : $('pr').value + ' мм';
  try {
    const [q, prof] = await Promise.all([api('/api/v1/forecast?' + params()), api('/api/v1/forecast?' + params({ agg: 'hour_profile' }))]);
    drawTS(q); drawProfile(prof); kpis(q, prof); ops(prof); drawMap();
  } catch (e) { toast(e.message); }
}

function drawTS(q) {
  const lab = q.data.map(d => d.hour !== undefined ? `${d.date.slice(5)} ${String(d.hour).padStart(2, '0')}h` : (d.date || d.month));
  const act = q.data.map(d => d.kind === 'actual' ? d.value : null), fc = q.data.map(d => d.kind === 'forecast' ? d.value : null);
  const lo = q.data.map(d => d.p10 ?? null), hi = q.data.map(d => d.p90 ?? null);
  $('ctitle').textContent = `Динамика: ${q.routes.length > 1 ? 'все маршруты' : 'маршрут ' + q.routes[0]}${q.stop ? ' · ' + q.stop.name : ''}${q.section ? ' · участок ' + q.section.from.name + ' — ' + q.section.to.name + ' (' + q.section.n_stops + ' ост.)' : ''} (${q.agg})`;
  const cfg = { type: q.agg === 'month' ? 'bar' : 'line', data: { labels: lab, datasets: [
    { label: 'Факт', data: act, borderColor: '#868e96', backgroundColor: '#868e9655', pointRadius: 0, borderWidth: 1.5 },
    { label: 'Прогноз', data: fc, borderColor: '#d9480f', backgroundColor: '#d9480f55', pointRadius: 0, borderWidth: 2 },
    ...(q.agg === 'month' ? [] : [
      { label: 'P90', data: hi, borderColor: 'transparent', backgroundColor: '#d9480f22', pointRadius: 0, fill: '+1' },
      { label: 'P10 (80%-интервал)', data: lo, borderColor: 'transparent', backgroundColor: '#d9480f22', pointRadius: 0, fill: false }])] },
    options: { animation: false, responsive: true, interaction: { mode: 'index', intersect: false }, scales: { y: { beginAtZero: true } },
      plugins: { tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y)}` } } } } };
  if (tsChart) tsChart.destroy(); tsChart = new Chart($('ts'), cfg);
}

function drawProfile(p) {
  const cfg = { type: 'bar', data: { labels: p.data.map(d => d.hour + 'h'), datasets: [{ label: 'Посадки/час', data: p.data.map(d => d.value), backgroundColor: '#1c7ed6' }] },
    options: { animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } } };
  if (profChart) profChart.destroy(); profChart = new Chart($('prof'), cfg);
}

function kpis(q, prof) {
  const tot = q.data.reduce((s, d) => s + d.value, 0);
  const days = new Set(q.data.map(d => d.date || d.month)).size;
  const pk = prof.data.reduce((a, b) => b.value > a.value ? b : a, { value: -1 });
  const fcShare = q.data.filter(d => d.kind === 'forecast').length / Math.max(q.data.length, 1);
  $('kpis').innerHTML = [
    ['Посадки за период', fmt(tot)], ['В среднем за ' + (q.agg === 'month' ? 'месяц' : 'день'), fmt(tot / Math.max(days, 1))],
    ['Час пик (ср. профиль)', `${pk.hour}:00 · ${fmt(pk.value)}`], ['Доля прогноза в выборке', Math.round(fcShare * 100) + '%']
  ].map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
}

function ops(prof) {
  // Little's law: mean passengers on board L = λ·W (λ = boardings per hour, W = mean ride time in hours)
  const r = $('route').value; if (r === 'all') { $('opsTable').innerHTML = '<p class="hint">Выберите конкретный маршрут.</p>'; return; }
  const sup = (ROUTES.find(x => x.route == r) || {}).supply || {};
  const weekendOnly = (() => { const a = new Date($('dfrom').value), b = new Date($('dto').value);
    for (let d = new Date(a); d <= b; d.setDate(d.getDate() + 1)) if (d.getDay() % 6 !== 0) return false; return true; })();
  const vh = weekendOnly ? sup.vehicles_by_hour_weekend_oct : sup.vehicles_by_hour_weekday_oct;
  const cap = +$('cap').value, tgt = +$('tgt').value / 100,
        W = +$('ride').value / 60, kp = +$('peak').value;
  const rows = prof.data.map(d => {
    const L = d.value * W;                              // average simultaneous passengers on the whole line
    const Lp = L * kp;                                  // peak-direction / uneven-headway allowance
    const need = Math.ceil(Lp / (cap * tgt));           // vehicles on line to keep load <= target
    const exits = vh ? vh[d.hour] : null;               // trams actually in service this hour (Oct median)
    const load = exits ? Lp / (exits * cap) : null;     // load with the actual (October) fleet
    const cls = load === null ? '' : load > 0.9 ? 'b-bad' : load > tgt ? 'b-warn' : 'b-ok';
    const delta = exits ? need - exits : null;
    return `<tr><td>${d.hour}:00</td><td>${fmt(d.value)}</td><td>${fmt(L)}</td><td>${need}</td><td>${exits ?? '—'}</td>` +
           `<td class="${cls}">${load === null ? '—' : Math.round(load * 100) + '%'}</td>` +
           `<td>${delta === null ? '—' : delta > 0 ? '+' + delta : delta}</td></tr>`;
  }).join('');
  $('opsTable').innerHTML = `<table><tr><th>Час</th><th>Посадки, чел/ч</th><th>В салонах одновременно (L=λ·W)</th><th>Нужно вагонов на линии</th><th>Вагонов на линии (факт, медиана окт.)</th><th>Наполняемость при текущем выпуске</th><th>Резерв/дефицит вагонов</th></tr>${rows}</table>
  <p class="hint">Закон Литтла: среднее число пассажиров в салонах = интенсивность посадок × среднее время поездки. Нужно вагонов = L·k<sub>пик</sub> / (вместимость·целевая наполняемость). «Вагонов на линии» — медиана числа разных вагонов (garage_number) с посадками в этот час, октябрь, будни (или выходные, если выбран только выходной период), из сырых валидаций; наполняемость — L·k<sub>пик</sub> / (вагоны·вместимость). Профиль — средний по выбранному периоду. k<sub>пик</sub> переводит среднюю по линии загрузку в загрузку максимального перегона в пиковом направлении (для радиальных маршрутов ≈ 2–3); значение требует калибровки по данным подсчёта пассажиров (АСКП).</p>`;
}

async function drawMap() {
  if (!map) {
    map = L.map('map').setView([55.76, 37.64], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18, attribution: '© OpenStreetMap' }).addTo(map);
  }
  if (layer) layer.remove(); layer = L.layerGroup().addTo(map);
  const h = +$('hour').value; $('hour_v').textContent = String(h).padStart(2, '0') + ':00';
  const sel = $('route').value, rs = sel === 'all' ? Object.keys(STOPS) : [sel];
  if (sel === 'all' && drawMap._fitted !== 'all') { map.setView([55.76, 37.64], 11); drawMap._fitted = 'all'; }
  const colors = ['#d9480f', '#1c7ed6', '#2b8a3e', '#ae3ec9', '#e67700', '#0c8599', '#c2255c'];
  const day = $('dfrom').value;
  for (const [k, r] of rs.entries()) {
    const s = STOPS[r]; if (!s) continue;
    let v = 0;
    try { const q = await api(`/api/v1/forecast?route=${r}&date_from=${day}&date_to=${day}&hour_from=${h}&hour_to=${h}&agg=total&k_weather=${$('kw').value}&k_event=${$('ke').value}&k_season=${$('ks').value}`); v = q.data[0].value; } catch (e) { continue; }
    const col = colors[k % colors.length], pts = s.stops.map(x => [x.lat, x.lon]);
    if (sel !== 'all' && drawMap._fitted !== sel) { map.invalidateSize(); map.fitBounds(L.latLngBounds(pts).pad(0.15)); drawMap._fitted = sel; }   // zoom to the chosen route
    L.polyline(pts, { color: col, weight: 3, opacity: .7 }).addTo(layer).bindTooltip(`Маршрут ${r}: ${fmt(v)} посадок в ${h}:00`);
    // highlight the selected section (участок)
    const ids = s.stops.map(x => String(x.stop_id)), a = ids.indexOf($('secA').value), b = ids.indexOf($('secB').value);
    if (sel !== 'all' && a >= 0 && b >= 0) L.polyline(pts.slice(Math.min(a, b), Math.max(a, b) + 1), { color: '#fab005', weight: 9, opacity: .6 }).addTo(layer).bindTooltip('Выбранный участок');
    s.stops.forEach(x => {
      const sh = x.share_by_hour ? x.share_by_hour[h] : x.share, val = v * sh;   // hour-dependent spatial share
      L.circleMarker([x.lat, x.lon], { radius: 2 + Math.sqrt(val) * 1.1, color: col, fillOpacity: .55, weight: 1 })
        .addTo(layer).bindTooltip(`${x.name}<br>маршрут ${r}, ${h}:00 — ≈${fmt(val)} посадок<br><span style="opacity:.7">доля ${(sh * 100).toFixed(1)} %; ${s.source}</span>`);
    });
  }
}

async function loadStops() {
  const r = $('route').value; $('stop').innerHTML = '<option value="">— весь маршрут —</option>';
  ['secA', 'secB'].forEach(id => $(id).innerHTML = '<option value="">—</option>');
  if (STOPS[r]) STOPS[r].stops.forEach(s => {
    const o = `<option value="${s.stop_id}">${s.seq}. ${s.name}</option>`;
    $('stop').insertAdjacentHTML('beforeend', o); $('secA').insertAdjacentHTML('beforeend', o); $('secB').insertAdjacentHTML('beforeend', o);
  });
}

async function init() {
  try {
    META = await api('/api/v1/meta'); ROUTES = await api('/api/v1/routes');
    $('meta').textContent = `модель ${META.model_version} · история ${META.first_day}…2025-10-31 · прогноз с ${META.forecast_start}`;
    $('route').innerHTML = '<option value="all">Все маршруты</option>' + ROUTES.map(r => `<option value="${r.route}">${r.route}${r.name ? ' · ' + r.name : ''}</option>`).join('');
    for (const r of ROUTES.filter(r => r.has_stops)) STOPS[r.route] = await api(`/api/v1/routes/${r.route}/stops`);
  } catch (e) { toast('Сервис недоступен: ' + e.message); return; }
  ['dfrom', 'dto', 'h1', 'h2', 'agg', 'kw', 'ke', 'ks', 'pr', 'stop', 'secA', 'secB'].forEach(id => $(id).addEventListener('change', refresh));
  $('stop').addEventListener('change', () => { if ($('stop').value) { $('secA').value = ''; $('secB').value = ''; } });
  ['secA', 'secB'].forEach(id => $(id).addEventListener('change', () => { if ($(id).value) $('stop').value = ''; }));
  ['kw', 'ke', 'ks', 'pr'].forEach(id => $(id).addEventListener('input', refresh));
  $('route').addEventListener('change', () => { loadStops(); refresh(); });
  $('horizon').addEventListener('change', onHorizon);
  $('hour').addEventListener('input', drawMap);
  ['cap', 'tgt', 'ride', 'peak'].forEach(id => $(id).addEventListener('change', refresh));
  $('reset').onclick = () => { ['kw', 'ke', 'ks'].forEach(k => $(k).value = 1); $('pr').value = -1; refresh(); };
  $('csv').onclick = () => location.href = '/api/v1/export?' + params({ format: 'csv' });
  $('xlsx').onclick = () => location.href = '/api/v1/export?' + params({ format: 'xlsx' });
  $('play').onclick = () => {
    if (playing) { clearInterval(playing); playing = null; $('play').textContent = '▶ Динамика'; return; }
    $('play').textContent = '⏸ Стоп'; playing = setInterval(() => { $('hour').value = (+$('hour').value + 1) % 24; drawMap(); }, 900);
  };
  refresh();
}
init();
