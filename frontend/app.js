(function () {
  "use strict";

  const state = {
    metricNames: {},   // key -> {name, unit, category, direction}
    pollTimer: null,
    currentMarket: null,
    lastData: null,
    expandedRow: null,
  };

  const el = (id) => document.getElementById(id);

  function riskClass(risk) {
    if (!risk) return "";
    const r = risk.toLowerCase();
    if (r === "low") return "score-low";
    if (r === "medium") return "score-medium";
    if (r === "high") return "score-high";
    return "score-extreme";
  }

  function showError(msg) {
    const line = el("error-line");
    line.textContent = msg;
    line.classList.remove("hidden");
  }

  function clearError() {
    el("error-line").classList.add("hidden");
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch (_) {}
      throw new Error(`${path} → ${res.status}: ${detail}`);
    }
    return res.json();
  }

  // ---------- Init ----------

  async function init() {
    checkHealth();
    showLoading("Loading markets…");
    await Promise.all([loadMarkets(), loadCoreMetrics()]);
    hideLoading();

    el("load-btn").addEventListener("click", onLoad);
    document.querySelectorAll(".admin-buttons button").forEach((btn) => {
      btn.addEventListener("click", () => onAdminAction(btn.dataset.action));
    });
  }

  async function checkHealth() {
    const dot = el("health-dot");
    const text = el("health-text");
    try {
      await api("/esg/health");
      dot.className = "dot up";
      text.textContent = "online";
    } catch (e) {
      dot.className = "dot down";
      text.textContent = "offline";
    }
  }

  async function loadMarkets() {
    try {
      const markets = await api("/esg/markets");
      const select = el("market-select");
      select.innerHTML = '<option value="">— select a market —</option>';

      const fetched = markets.filter((m) => m.company_count > 0).sort((a, b) => a.name.localeCompare(b.name));
      const notFetched = markets.filter((m) => m.company_count === 0).sort((a, b) => a.name.localeCompare(b.name));

      const addGroup = (label, list) => {
        if (!list.length) return;
        const group = document.createElement("optgroup");
        group.label = label;
        for (const m of list) {
          const opt = document.createElement("option");
          opt.value = m.name;
          opt.textContent = `${m.name} (${m.company_count})`;
          group.appendChild(opt);
        }
        select.appendChild(group);
      };

      addGroup("Fetched", fetched);
      addGroup("Not Fetched", notFetched);
    } catch (e) {
      showError("Failed to load markets: " + e.message);
    }
  }

  async function loadCoreMetrics() {
    try {
      const metrics = await api("/esg/core-metrics");
      for (const m of metrics) {
        state.metricNames[m.key] = m;
      }
    } catch (e) {
      showError("Failed to load core metrics: " + e.message);
    }
  }

  // ---------- Load / poll ----------

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function showLoading(text) {
    el("loading-text").textContent = text;
    el("loading-line").classList.remove("hidden");
  }

  function hideLoading() {
    el("loading-line").classList.add("hidden");
  }

  async function onLoad() {
    const typed = el("market-input").value.trim();
    const selected = el("market-select").value;
    const marketName = typed || selected;
    if (!marketName) {
      showError("Pick or type a market name first.");
      return;
    }
    clearError();
    stopPolling();
    state.currentMarket = marketName;
    state.expandedRow = null;

    const loadBtn = el("load-btn");
    loadBtn.disabled = true;
    showLoading(`Fetching ESG data for "${marketName}"…`);

    try {
      const data = await api("/esg/market-esg", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ market_name: marketName }),
      });
      render(data);
      maybeStartPolling(marketName, data);
    } catch (e) {
      showError(e.message);
    } finally {
      loadBtn.disabled = false;
      hideLoading();
    }
  }

  function maybeStartPolling(marketName, data) {
    const needsPoll = data.companies.some(
      (c) => c.esg_scoring === "pending" || c.esg_scoring === "processing" || !c.has_revenue
    );
    if (!needsPoll) {
      el("status-strip").classList.add("hidden");
      return;
    }
    el("status-strip").classList.remove("hidden");
    pollStatus(marketName);
    state.pollTimer = setInterval(() => pollStatus(marketName), 5000);
  }

  async function pollStatus(marketName) {
    if (marketName !== state.currentMarket) return;
    try {
      const status = await api("/esg/estimation-status", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ market_name: marketName }),
      });
      if (marketName !== state.currentMarket) return;
      renderStatus(status);
      if (status.all_ready) {
        stopPolling();
        const data = await api("/esg/market-esg", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ market_name: marketName }),
        });
        if (marketName === state.currentMarket) {
          render(data);
          el("status-strip").classList.add("hidden");
        }
      }
    } catch (e) {
      showError("Polling failed: " + e.message);
    }
  }

  function renderStatus(status) {
    el("status-text").textContent =
      `${status.done}/${status.total_companies} ready · ${status.processing} processing · ${status.pending} pending`;
    const pct = status.total_companies ? Math.round((status.done / status.total_companies) * 100) : 0;
    el("progress-fill").style.width = pct + "%";

    const chips = el("status-chips");
    chips.innerHTML = "";
    for (const c of status.companies) {
      const chip = document.createElement("span");
      chip.className = "chip " + c.state;
      chip.textContent = `${c.name} · ${c.state}`;
      chips.appendChild(chip);
    }
  }

  // ---------- Render ----------

  function render(data) {
    state.lastData = data;

    el("summary-row").classList.remove("hidden");
    el("summary-avg").textContent = data.industry_avg_esg_score;
    el("summary-total").textContent = data.total_companies;
    el("summary-method").textContent = data.scoring_method;

    const table = el("company-table");
    table.classList.remove("hidden");
    const tbody = el("company-tbody");
    tbody.innerHTML = "";

    data.companies.forEach((c, idx) => {
      const row = document.createElement("tr");
      row.className = "company-row";
      row.innerHTML = `
        <td>${idx + 1}</td>
        <td>${escapeHtml(c.name)}</td>
        <td>${escapeHtml(c.country || "—")}</td>
        <td class="${riskClass(c.esg_scores.environment.risk)}">${fmtScore(c.esg_scores.environment.score)}</td>
        <td class="${riskClass(c.esg_scores.social.risk)}">${fmtScore(c.esg_scores.social.score)}</td>
        <td class="${riskClass(c.esg_scores.governance.risk)}">${fmtScore(c.esg_scores.governance.score)}</td>
        <td class="${riskClass(c.esg_scores.total.risk)}">${fmtScore(c.esg_scores.total.score)}</td>
        <td><span class="badge">${escapeHtml(c.rating || "—")}</span></td>
        <td><span class="badge">${escapeHtml(c.data_source || "—")}</span></td>
        <td>${c.metrics_disclosed}</td>
        <td>${c.metrics_estimated}</td>
      `;
      row.addEventListener("click", () => toggleDetail(row, c));
      tbody.appendChild(row);
    });
  }

  function fmtScore(s) {
    return s === null || s === undefined ? "—" : s;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  function toggleDetail(row, company) {
    const next = row.nextElementSibling;
    if (next && next.classList.contains("detail-row")) {
      next.remove();
      return;
    }
    document.querySelectorAll(".detail-row").forEach((n) => n.remove());

    const detail = document.createElement("tr");
    detail.className = "detail-row";
    const td = document.createElement("td");
    td.colSpan = 11;
    td.appendChild(buildPillarBlock("Environmental", company.environmental_metrics));
    td.appendChild(buildPillarBlock("Social", company.social_metrics));
    td.appendChild(buildPillarBlock("Governance", company.governance_metrics));
    detail.appendChild(td);
    row.after(detail);
  }

  function buildPillarBlock(title, metrics) {
    const block = document.createElement("div");
    block.className = "pillar-block";
    const h4 = document.createElement("h4");
    h4.textContent = title;
    block.appendChild(h4);

    if (!metrics || Object.keys(metrics).length === 0) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = "No metrics available yet.";
      block.appendChild(p);
      return block;
    }

    const table = document.createElement("table");
    table.className = "metric-table";
    table.innerHTML = `
      <thead><tr><th>Metric</th><th>Value</th><th>Score</th><th>Year</th><th>Provenance</th></tr></thead>
    `;
    const tbody = document.createElement("tbody");
    for (const [key, m] of Object.entries(metrics)) {
      const meta = state.metricNames[key];
      const label = meta ? meta.name : prettify(key);
      const tr = document.createElement("tr");
      let prov = "";
      if (m.corrected) {
        prov = '<span class="prov-badge corrected">corrected</span>';
      } else if (m.estimated) {
        prov = `<span class="prov-badge">est${m.confidence != null ? " " + m.confidence : ""}</span>`;
      }
      tr.innerHTML = `
        <td>${escapeHtml(label)}</td>
        <td>${escapeHtml(m.value)}</td>
        <td>${fmtScore(m.score)}</td>
        <td>${m.year || "—"}</td>
        <td>${prov}</td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    block.appendChild(table);
    return block;
  }

  function prettify(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // ---------- Admin panel ----------

  async function onAdminAction(action) {
    const log = el("admin-log");
    const marketName = el("market-input").value.trim() || el("market-select").value;
    const write = (obj) => {
      log.textContent = JSON.stringify(obj, null, 2) + "\n\n" + log.textContent;
    };

    const btn = document.querySelector(`.admin-buttons button[data-action="${action}"]`);
    const originalLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Running…";

    try {
      let result;
      switch (action) {
        case "get-players":
          if (!marketName) return write({ error: "Type or select a market name first." });
          result = await api("/key_players/get_players_by_market", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ market_name: marketName }),
          });
          break;
        case "fetch-market":
          if (!marketName) return write({ error: "Type or select a market name first." });
          result = await api("/esg/fetch-market", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ market_name: marketName }),
          });
          break;
        case "seed-metrics":
          result = await api("/esg/seed-metrics", { method: "POST" });
          break;
        case "seed-market-metrics":
          result = await api("/esg/seed-market-metrics", { method: "POST" });
          break;
        case "seed-markets":
          result = await api("/key_players/seed-markets", { method: "POST" });
          break;
        case "start-pipeline":
          result = await api("/key_players/start-pipeline", { method: "POST" });
          break;
      }
      write(result);
      if (action === "seed-markets" || action === "get-players") loadMarkets();
    } catch (e) {
      write({ error: e.message });
    } finally {
      btn.disabled = false;
      btn.textContent = originalLabel;
    }
  }

  init();
})();
