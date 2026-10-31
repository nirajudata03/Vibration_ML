import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, AlertTriangle, ArrowDownUp, CheckCircle2, ChevronDown, CircleHelp, CloudUpload, Cpu, Download, FileText, FolderOpen, Gauge, HardDrive, Play, Power, Radio, RefreshCw, Search, Square, Wifi, WifiOff } from 'lucide-react';

const API = '/api';
const fmt = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const faultColor = (fault = '') => fault.toLowerCase() === 'normal' ? 'green' : 'orange';
const countdown = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
const fileSize = (bytes) => bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

export default function App() {
  const [live, setLive] = useState(null);
  const [session, setSession] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [activating, setActivating] = useState(false);
  const [loggerStarting, setLoggerStarting] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [files, setFiles] = useState([]);
  const [fileFolder, setFileFolder] = useState('all');
  const [rows, setRows] = useState([]);
  const [filename, setFilename] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('All faults');
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState('row');
  const [sortUp, setSortUp] = useState(true);
  const [label, setLabel] = useState('Normal');
  const [duration, setDuration] = useState(2);
  const inputRef = useRef(null);
  const wasRecording = useRef(false);

  const refreshFiles = useCallback(async () => {
    try {
      const response = await fetch(`${API}/files`);
      if (response.ok) setFiles(await response.json());
    } catch { /* Files appear when the backend is online. */ }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [healthRes, dataRes, sessionRes] = await Promise.all([fetch(`${API}/health`), fetch(`${API}/logger/data`), fetch(`${API}/logger/session/status`)]);
      setBackendStatus(healthRes.ok ? 'online' : 'offline');
      if (dataRes.ok) { setLive(await dataRes.json()); setLoggerStarting(false); }
      if (sessionRes.ok) {
        const nextSession = await sessionRes.json();
        if (wasRecording.current && !nextSession.recording && nextSession.csv_path) {
          setMessage(`Recording complete. Saved to data/readings/${nextSession.csv_path.split(/[\\/]/).pop()}`);
          refreshFiles();
        }
        wasRecording.current = nextSession.recording;
        setSession(nextSession);
      }
    } catch { setBackendStatus('offline'); }
  }, [refreshFiles]);

  useEffect(() => {
    refresh();
    refreshFiles();
    const timer = window.setInterval(refresh, 1200);
    return () => window.clearInterval(timer);
  }, [refresh, refreshFiles]);

  const visibleRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return rows.filter((row) => (filter === 'All faults' || row.predicted_fault === filter)
      && (!normalized || String(row.row).includes(normalized) || String(row.predicted_fault).toLowerCase().includes(normalized)))
      .slice().sort((a, b) => {
        const left = a[sortKey]; const right = b[sortKey];
        const compared = typeof left === 'number' && typeof right === 'number' ? left - right : String(left ?? '').localeCompare(String(right ?? ''));
        return sortUp ? compared : -compared;
      });
  }, [rows, filter, query, sortKey, sortUp]);

  const counts = useMemo(() => rows.reduce((all, row) => {
    all[row.predicted_fault] = (all[row.predicted_fault] || 0) + 1;
    return all;
  }, {}), [rows]);

  async function predictFile(file) {
    if (!file) return;
    setBusy(true); setError(''); setMessage('Uploading your CSV and running the model…'); setRows([]); setFilename(file.name);
    try {
      const form = new FormData(); form.append('file', file);
      const response = await fetch(`${API}/predict`, { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not predict this file.');
      setRows(result.rows); setMessage(`Prediction complete · ${result.rows.length.toLocaleString()} rows · saved to data/predictions/${result.output_file}`); refreshFiles();
    } catch (err) { setError(err.message); setMessage(''); }
    finally { setBusy(false); if (inputRef.current) inputRef.current.value = ''; }
  }

  async function predictSavedFile(file) {
    setBusy(true); setError(''); setMessage(`Loading ${file.name} into the model…`); setRows([]); setFilename(file.name);
    try {
      const response = await fetch(`${API}/predict-saved`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: file.folder, name: file.name }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not predict this saved file.');
      setRows(result.rows);
      setMessage(`Prediction complete · ${result.rows.length.toLocaleString()} rows · saved to data/predictions/${result.output_file}`);
      refreshFiles();
      document.getElementById('predictions')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) { setError(err.message); setMessage(''); }
    finally { setBusy(false); }
  }

  async function sessionAction(action) {
    setError(''); setMessage('');
    try {
      const response = await fetch(`${API}/logger/session/${action}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: action === 'start' ? JSON.stringify({ label, duration_min: duration }) : '{}',
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Logger request failed.');
      setMessage(action === 'start' ? `Recording ${result.label} readings now.` : `Recording saved with ${result.row_count} rows to data/readings/${(result.csv_path || 'CSV file').split(/[\\\\/]/).pop()}.`);
      refresh();
      refreshFiles();
    } catch (err) { setError(err.message); }
  }

  async function activateBackend() {
    setActivating(true); setError(''); setMessage('');
    try {
      const response = await fetch(`${API}/logger/activate`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not activate sensor backend.');
      setLoggerStarting(Boolean(result.starting));
      setMessage(result.message);
      setTimeout(refresh, 1200);
    } catch (err) { setError(err.message); }
    finally { setActivating(false); }
  }

  async function stopServers() {
    setMenuOpen(false);
    try {
      await fetch(`${API}/shutdown`, { method: 'POST' });
    } catch { /* Continue and stop Vite even if the API is already down. */ }
    try {
      await fetch('/__local/stop', { method: 'POST' });
    } catch {
      setError('Could not stop the Vite server from the page. Stop it with Ctrl+C in its terminal.');
    }
  }

  function downloadCsv() {
    if (!rows.length) return;
    const headers = Object.keys(rows[0]);
    const csvCell = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const csv = [headers.map(csvCell).join(','), ...rows.map((row) => headers.map((key) => csvCell(row[key])).join(','))].join('\r\n');
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    link.download = `${filename.replace(/\.csv$/i, '') || 'readings'}_predictions.csv`; link.click(); URL.revokeObjectURL(link.href);
  }

  const connected = live?.serial_connected === true;
  const isRecording = session?.recording === true;
  const remainingSeconds = Math.max(0, Number(session?.remaining_s ?? 0));
  const visibleFiles = files.filter((file) => fileFolder === 'all' || file.folder === fileFolder);
  const toggleSort = (key) => { if (sortKey === key) setSortUp(!sortUp); else { setSortKey(key); setSortUp(true); } };

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Activity size={21} /></div><div><b>VibeGuard</b><span>MONITORING SYSTEM</span></div></div>
      <div className="side-label">WORKSPACE</div>
      <a className="nav-item active" href="#overview"><Gauge size={17} /> Overview <span className="nav-dot" /></a>
      <a className="nav-item" href="#predictions"><FileText size={17} /> Predictions</a>
      <a className="nav-item" href="#sensor"><Radio size={17} /> Live sensor</a>
      <a className="nav-item" href="#data-files"><FolderOpen size={17} /> Data files</a>
      <div className="side-spacer" />
      <div className="model-card"><div className="model-icon"><Cpu size={17} /></div><div><b>Random Forest</b><span>Model ready</span></div><CheckCircle2 className="model-check" size={16} /></div>
      <div className="sidebar-foot"><span className="pulse-dot" /> Local machine <span className="local-tag">LOCAL</span></div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumb">Vibration ML <span>/</span> <b>Overview</b></div><div className="top-actions"><div className={`connection ${connected ? 'online' : ''}`}><span className="pulse-dot" />{connected ? 'Sensor connected' : 'Sensor offline'}</div><button className={`backend-button ${backendStatus}`} onClick={activateBackend} disabled={activating || loggerStarting || backendStatus === 'offline'} title={backendStatus === 'offline' ? 'Start the Python backend first' : 'Start the sensor logger if it is not active'}><Radio size={14} />{backendStatus === 'offline' ? 'Backend offline' : loggerStarting ? 'Starting…' : activating ? 'Activating…' : 'Activate backend'}</button><button className="icon-button" title="Refresh sensor status" onClick={() => { refresh(); refreshFiles(); }}><RefreshCw size={16} /></button><div className="avatar-menu-wrap"><button className="avatar" aria-haspopup="menu" aria-expanded={menuOpen} aria-label="Open account and server menu" onClick={() => setMenuOpen(!menuOpen)}>N<ChevronDown size={11} /></button>{menuOpen && <div className="avatar-menu" role="menu"><div className="menu-heading"><b>VibeGuard</b><span>Local project controls</span></div><button role="menuitem" className="stop-services-item" onClick={stopServers}><Power size={15} /><span><b>Stop all servers</b><small>Stop React, Python API, and logger</small></span></button></div>}</div></div></header>
      <div className="page-content" id="overview">
        <section className="welcome-row"><div><div className="eyebrow">MOTOR HEALTH MONITORING</div><h1>Dashboard overview</h1><p>Review live vibration readings and inspect fault predictions from your CSV data.</p></div><div className="updated"><span className="pulse-dot" /> Live updates <span className="separator">·</span> every 1.2 sec</div></section>

        <section className="stat-grid">
          <article className="stat-card"><div className="stat-top"><span>CONNECTION</span><div className={`stat-icon ${connected ? 'mint' : 'muted'}`}>{connected ? <Wifi size={17} /> : <WifiOff size={17} />}</div></div><div className="stat-value small-value">{connected ? 'Connected' : 'Waiting'}</div><div className="stat-note">{connected ? 'Receiving sensor data' : live?.serial_error || 'Check the logger and sensor COM port'}</div></article>
          <article className="stat-card"><div className="stat-top"><span>RECORDING</span><div className={`stat-icon ${isRecording ? 'amber' : 'muted'}`}><Radio size={17} /></div></div><div className="stat-value small-value">{isRecording ? 'In progress' : 'Not recording'}</div><div className="stat-note">{isRecording ? `${session.row_count || 0} readings collected` : 'Start a collection session below'}</div></article>
          <article className="stat-card"><div className="stat-top"><span>ROWS PREDICTED</span><div className="stat-icon blue"><Activity size={17} /></div></div><div className="stat-value">{rows.length.toLocaleString()}</div><div className="stat-note">{filename || 'Upload a CSV file to begin'}</div></article>
          <article className="stat-card fault-stat-card"><div className="stat-top"><span>FAULT TYPES</span><div className="stat-icon violet"><AlertTriangle size={17} /></div></div><div className="fault-name-list">{Object.keys(counts).length ? Object.keys(counts).map((fault) => <span className={`fault-pill ${faultColor(fault)}`} key={fault}><span />{fault}</span>) : <span className="stat-note">Upload data to see fault names</span>}</div><div className="stat-note">Names detected in the current prediction</div></article>
        </section>

        <section className="content-grid">
          <article className="panel upload-panel"><div className="panel-heading"><div><h2>Predict from a CSV</h2><p>Choose a readings file and run your trained model.</p></div><div className="heading-icon"><CloudUpload size={18} /></div></div>
            <button className={`drop-zone ${busy ? 'is-busy' : ''}`} onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); predictFile(event.dataTransfer.files[0]); }} disabled={busy}>
              <input ref={inputRef} type="file" accept=".csv,text/csv" hidden onChange={(event) => predictFile(event.target.files[0])} />
              <span className="upload-icon">{busy ? <RefreshCw size={20} className="spin" /> : <CloudUpload size={21} />}</span>
              <b>{busy ? 'Running prediction…' : 'Click to choose a CSV'}</b><span>or drag and drop it here</span><small>CSV file · includes the 18 vibration feature columns</small>
            </button>
            <div className="upload-footer"><span><HardDrive size={14} /> Saved in <code>data/predictions</code></span><span className="secure-note"><CheckCircle2 size={14} /> Checked before prediction</span></div>
            {message && <div className="notice success"><CheckCircle2 size={15} />{message}</div>}{error && <div className="notice failure"><AlertTriangle size={15} />{error}</div>}
          </article>

          <article className="panel live-panel" id="sensor"><div className="panel-heading"><div><h2>Live vibration</h2><p>Latest values from the connected logger.</p></div><div className={`live-badge ${connected ? '' : 'offline'}`}><span className="pulse-dot" />{connected ? 'LIVE' : 'OFFLINE'}</div></div>
            <div className="axis-grid">{['x', 'y', 'z'].map((axis) => <div className="axis-card" key={axis}><div className="axis-head"><span className={`axis-dot axis-${axis}`} />AXIS {axis.toUpperCase()}</div><div className="axis-reading">{fmt(live?.[`rms_${axis}`], 4)}<small>g</small></div><div className="axis-foot">Peak <b>{fmt(live?.[`peak_${axis}`], 3)} g</b></div></div>)}</div>
            <div className="live-meta"><span>Sample rate <b>{fmt(live?.actual_fs, 0)} Hz</b></span><span>Cycle <b>{live?.cycle ?? '—'}</b></span><span>I²C errors <b>{live?.i2c_errors ?? '—'}</b></span></div>
          </article>
        </section>

        <section className="panel recording-panel"><div className="recording-copy"><div className="recording-symbol"><Radio size={19} /></div><div><h2>Reading collection</h2><p>Start a labeled recording using the sensor logger.</p></div>{isRecording && <div className="recording-timer"><span>TIME LEFT</span><b>{countdown(remainingSeconds)}</b></div>}</div><div className="recording-controls"><label className="select-wrap"><span>FAULT LABEL</span><select value={label} onChange={(event) => setLabel(event.target.value)} disabled={isRecording}><option>Normal</option><option>Imbalance</option><option>Misalignment</option><option>Looseness</option></select><ChevronDown size={14} /></label><label className="duration-wrap"><span>DURATION</span><div><input type="number" min="0.1" max="120" step="0.1" value={duration} onChange={(event) => setDuration(event.target.value)} disabled={isRecording} /><small>min</small></div></label>{isRecording ? <button className="stop-button" onClick={() => sessionAction('stop')}><Square size={14} fill="currentColor" /> Stop recording</button> : <button className="start-button" onClick={() => sessionAction('start')}><Play size={14} fill="currentColor" /> Start recording</button>}</div></section>

        <section className="panel predictions-panel" id="predictions"><div className="results-heading"><div><div className="section-kicker">MODEL OUTPUT</div><h2>Prediction results <span className="result-count">{rows.length.toLocaleString()}</span></h2><p>{filename ? `Showing results for ${filename}` : 'Upload a CSV to see predicted faults here.'}</p></div>{rows.length > 0 && <button className="download-button" onClick={downloadCsv}><Download size={15} /> Download CSV</button>}</div>
          <div className="table-tools"><label className="search-box"><Search size={15} /><input placeholder="Search row or fault…" value={query} onChange={(event) => setQuery(event.target.value)} /></label><label className="filter-box"><span>Show:</span><select value={filter} onChange={(event) => setFilter(event.target.value)}><option>All faults</option>{[...new Set(rows.map((row) => row.predicted_fault))].map((fault) => <option key={fault}>{fault}</option>)}</select><ChevronDown size={14} /></label></div>
          <div className="table-wrap"><table><thead><tr><th><button onClick={() => toggleSort('row')}>ROW <ArrowDownUp size={12} /></button></th><th><button onClick={() => toggleSort('predicted_fault')}>PREDICTED FAULT <ArrowDownUp size={12} /></button></th>{rows[0] && Object.keys(rows[0]).filter((key) => key.startsWith('probability_')).map((key) => <th key={key}><button onClick={() => toggleSort(key)}>{key.replace('probability_', '').toUpperCase()} <ArrowDownUp size={12} /></button></th>)}</tr></thead><tbody>{visibleRows.slice(0, 250).map((row) => <tr key={row.row}><td className="row-index">{row.row}</td><td><span className={`fault-pill ${faultColor(row.predicted_fault)}`}><span />{row.predicted_fault}</span></td>{Object.keys(row).filter((key) => key.startsWith('probability_')).map((key) => <td className="probability" key={key}>{(Number(row[key]) * 100).toFixed(1)}%</td>)}</tr>)}</tbody></table>
            {!rows.length && <div className="empty-state"><div className="empty-icon"><FileText size={21} /></div><b>No prediction data yet</b><span>Upload a CSV above. Your predictions will appear in this table.</span></div>}
            {visibleRows.length > 250 && <div className="table-limit">Showing first 250 of {visibleRows.length.toLocaleString()} rows. Download CSV to view all rows.</div>}
          </div>
          <div className="table-foot"><span><span className="tiny-green" /> Model output</span><span>Prediction results are saved locally on this computer.</span></div>
        </section>
        <section className="panel files-panel" id="data-files"><div className="results-heading"><div><div className="section-kicker">LOCAL STORAGE</div><h2>Data folder files <span className="result-count">{visibleFiles.length}</span></h2><p>Browse and download CSV files saved by the logger, uploads, and model.</p></div><button className="download-button" onClick={refreshFiles}><RefreshCw size={14} /> Refresh list</button></div><div className="file-filters">{[['all', 'All files'], ['readings', 'Readings'], ['uploads', 'Uploads'], ['predictions', 'Predictions']].map(([key, title]) => <button className={fileFolder === key ? 'selected' : ''} key={key} onClick={() => setFileFolder(key)}>{title}</button>)}</div><div className="file-list">{visibleFiles.length ? visibleFiles.map((file) => <div className="file-row" key={`${file.folder}/${file.name}`}><div className="file-type-icon"><FileText size={16} /></div><div className="file-name"><b>{file.name}</b><span>data/{file.folder} · {file.modified}</span></div><span className="file-size">{fileSize(file.bytes)}</span><button className="file-predict" onClick={() => predictSavedFile(file)} disabled={busy} title="Run the model on this CSV">{busy && filename === file.name ? <RefreshCw size={14} className="spin" /> : <Cpu size={14} />}<span>Load to prediction</span></button><a className="file-download" href={`${API}/files/${file.folder}/${encodeURIComponent(file.name)}`} download title="Download CSV"><Download size={15} /></a></div>) : <div className="files-empty">No CSV files in this folder yet.</div>}</div></section>
        <footer className="page-foot"><span>VibeGuard <span>·</span> Vibration Monitoring</span><span className="footer-help"><CircleHelp size={14} /> Local dashboard for your project</span></footer>
      </div>
    </main>
  </div>;
}

