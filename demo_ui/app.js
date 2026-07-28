(function () {
  "use strict";

  var DATA = window.DEMO_DATA || { total_companies_in_db: 0, markets: [] };
  var marketSelect = document.getElementById("marketSelect");
  var runBtn = document.getElementById("runBtn");
  var dbNote = document.getElementById("dbNote");
  var marketSummary = document.getElementById("marketSummary");
  var companyGrid = document.getElementById("companyGrid");
  var detailPanel = document.getElementById("detailPanel");
  var stageRail = document.getElementById("stageRail");

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function riskClass(risk) {
    return "risk-" + String(risk || "").toLowerCase();
  }

  // c.esg_scoring (companies.esg_scoring): pending | processing | estimated | reported
  function scoringBadgeClass(status) {
    var known = ["estimated", "reported", "processing", "pending"];
    return known.indexOf(status) >= 0 ? status : "pending";
  }

  function scoringBadgeLabel(status) {
    var map = {
      estimated: "Estimated", reported: "Reported",
      processing: "Processing", pending: "Pending",
    };
    return map[status] || status || "Unknown";
  }

  // c.data_source (build_esg_json.py's per-company provenance):
  // reported | mixed | agentic_estimated | no_data
  function dataSourceClass(status) {
    var known = { reported: "reported", mixed: "mixed", agentic_estimated: "estimated", no_data: "no_data" };
    return known[status] || "no_data";
  }

  function dataSourceLabel(status) {
    var map = {
      reported: "Reported", mixed: "Mixed",
      agentic_estimated: "Estimated", no_data: "No data",
    };
    return map[status] || status || "Unknown";
  }

  // ── Rendering ────────────────────────────────────────────────────────────

  function populateMarketSelect() {
    marketSelect.innerHTML = "";
    DATA.markets.forEach(function (m, i) {
      var opt = document.createElement("option");
      opt.value = i;
      opt.textContent = m.market + " (" + m.total_companies + " companies)";
      marketSelect.appendChild(opt);
    });
  }

  function renderDbNote() {
    dbNote.textContent =
      "Checking database of " + DATA.total_companies_in_db.toLocaleString() +
      " companies for existing ESG data before estimating gaps.";
  }

  function renderMarketSummary(market) {
    marketSummary.innerHTML =
      statBlock("Companies", market.total_companies) +
      statBlock("Industry Avg ESG Score", market.industry_avg_esg_score) +
      statBlock("Market Status", market.status || "n/a");
  }

  function statBlock(label, value) {
    return (
      '<div class="summary-stat">' +
        '<div class="label">' + esc(label) + '</div>' +
        '<div class="value">' + esc(value) + '</div>' +
      '</div>'
    );
  }

  // A company with data_source 'no_data' has no real or estimated metrics at
  // all -- build_esg_json.py still emits a 0.0/"Extreme" score object for it
  // (there's nothing to average), which reads as "worst possible ESG score"
  // rather than "not yet processed". The UI treats no_data as its own state
  // instead of showing those placeholder numbers as if they were real.
  function hasNoData(c) {
    return c.data_source === "no_data";
  }

  function renderCompanyGrid(market) {
    companyGrid.innerHTML = "";
    market.companies.forEach(function (c, idx) {
      var card = document.createElement("div");
      card.className = "company-card";
      card.dataset.idx = idx;
      card.innerHTML =
        '<div class="name">' + esc(c.name) + '</div>' +
        '<div class="country">' + esc(c.country || "Unknown location") + '</div>' +
        '<div class="score-row"></div>';
      card.addEventListener("click", function () {
        selectCompany(market, idx, card);
      });
      companyGrid.appendChild(card);
    });
  }

  // Fills in the badge + score chips on already-rendered cards, with a fade-
  // in transition -- called once scoring is "done" in the staged reveal.
  function revealScores(market) {
    var cards = companyGrid.querySelectorAll(".company-card");
    Array.prototype.forEach.call(cards, function (card) {
      var idx = Number(card.dataset.idx);
      var c = market.companies[idx];
      var status = scoringBadgeClass(c.esg_scoring);
      var noData = hasNoData(c);
      var badge = document.createElement("span");
      badge.className = "badge " + status;
      badge.textContent = scoringBadgeLabel(status);
      card.appendChild(badge);

      var scoreRow = card.querySelector(".score-row");
      scoreRow.innerHTML = noData
        ? '<span class="score-chip pending-chip">Not yet processed</span>'
        : scoreChip("E", c.esg_scores.environment.score) +
          scoreChip("S", c.esg_scores.social.score) +
          scoreChip("G", c.esg_scores.governance.score) +
          scoreChip("Total", c.esg_scores.total.score);
      // rAF so the opacity/transform transition actually plays instead of
      // the class landing on the same frame the innerHTML is set.
      requestAnimationFrame(function () {
        scoreRow.classList.add("visible");
      });
    });
  }

  function scoreChip(label, score) {
    return '<span class="score-chip">' + label + ": " + esc(score) + '</span>';
  }

  function selectCompany(market, idx, cardEl) {
    var wasSelected = cardEl.classList.contains("selected");
    Array.prototype.forEach.call(
      companyGrid.querySelectorAll(".company-card"),
      function (el) { el.classList.remove("selected"); }
    );
    if (wasSelected) {
      detailPanel.classList.remove("open");
      detailPanel.innerHTML = "";
      return;
    }
    cardEl.classList.add("selected");
    renderDetail(market.companies[idx]);
    detailPanel.classList.add("open");
    detailPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderDetail(c) {
    detailPanel.innerHTML =
      '<div class="detail-header">' +
        '<div>' +
          '<h2>' + esc(c.name) + '</h2>' +
          '<div class="sub">' + esc(c.country || "Unknown location") +
            ' &middot; Data source: ' + esc(dataSourceLabel(c.data_source)) +
            (c.reporting_year ? ' &middot; Reporting year ' + esc(c.reporting_year) : '') +
          '</div>' +
        '</div>' +
        '<button class="close-btn" id="closeDetailBtn">Close</button>' +
      '</div>' +
      (hasNoData(c)
        ? '<div class="section"><div class="no-data">This company has not been processed yet — no scores computed.</div></div>'
        : pillarScoresBlock(c) +
          reasoningSection(c) +
          summarySection(c) +
          metricsSection("Environmental Metrics", c.environmental_metrics) +
          metricsSection("Social Metrics", c.social_metrics) +
          metricsSection("Governance Metrics", c.governance_metrics));

    document.getElementById("closeDetailBtn").addEventListener("click", function () {
      detailPanel.classList.remove("open");
      detailPanel.innerHTML = "";
      Array.prototype.forEach.call(
        companyGrid.querySelectorAll(".company-card"),
        function (el) { el.classList.remove("selected"); }
      );
    });
  }

  function pillarScoresBlock(c) {
    var s = c.esg_scores;
    return (
      '<div class="pillar-scores">' +
        pillarCard("Environment", s.environment) +
        pillarCard("Social", s.social) +
        pillarCard("Governance", s.governance) +
        pillarCard("Total", s.total) +
      '</div>'
    );
  }

  function pillarCard(label, scoreObj) {
    return (
      '<div class="pillar-card">' +
        '<div class="label">' + esc(label) + '</div>' +
        '<div class="score">' + esc(scoreObj.score) + '</div>' +
        '<div class="risk ' + riskClass(scoreObj.risk) + '">' + esc(scoreObj.risk) + ' risk</div>' +
      '</div>'
    );
  }

  function reasoningSection(c) {
    var r = c.reasoning || {};
    var parts = [];
    if (r.environment) parts.push(reasoningBlock("E", r.environment));
    if (r.social) parts.push(reasoningBlock("S", r.social));
    if (r.governance) parts.push(reasoningBlock("G", r.governance));
    if (!parts.length) return "";
    return (
      '<div class="section"><h3>Pillar Reasoning</h3>' + parts.join("") + '</div>'
    );
  }

  function reasoningBlock(tag, text) {
    return (
      '<div class="reasoning-block"><span class="pillar-tag">' + tag + ':</span>' +
      esc(text) + '</div>'
    );
  }

  function summarySection(c) {
    if (!c.summary) return "";
    return (
      '<div class="section"><h3>Summary</h3>' +
      '<div class="summary-text">' + esc(c.summary) + '</div></div>'
    );
  }

  function metricsSection(title, metrics) {
    if (!metrics) return "";
    var keys = Object.keys(metrics);
    if (!keys.length) return "";
    var rows = keys.map(function (k) {
      var m = metrics[k];
      var pill = m.estimated
        ? '<span class="pill-small est">Estimated</span>'
        : '<span class="pill-small real">Reported</span>';
      var conf = (m.estimated && m.confidence !== undefined && m.confidence !== null)
        ? (Math.round(m.confidence * 100) + "%") : "&mdash;";
      return (
        '<tr>' +
          '<td>' + esc(prettifyKey(k)) + '</td>' +
          '<td>' + esc(formatMetricValue(m)) + '</td>' +
          '<td>' + esc(m.year || "&mdash;") + '</td>' +
          '<td>' + pill + '</td>' +
          '<td>' + conf + '</td>' +
        '</tr>'
      );
    }).join("");

    return (
      '<div class="section"><h3>' + esc(title) + '</h3>' +
      '<table class="metrics-table">' +
        '<thead><tr><th>Metric</th><th>Value</th><th>Year</th><th>Source</th><th>Confidence</th></tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table></div>'
    );
  }

  function formatMetricValue(m) {
    // value strings from build_esg_json already embed the unit
    // ("15,000 tCO2e", "9,000 #", "Yes" with unit "Yes/No"), so only
    // append the unit when the value doesn't already carry it.
    var val = (m.value === null || m.value === undefined) ? "—" : String(m.value);
    var unit = m.unit ? String(m.unit) : "";
    val = val.replace(/\s*#\s*$/, "");            // "9,000 #" -> "9,000"
    if (!unit || unit === "#" || unit === "Yes/No") return val;
    if (val.indexOf(unit) !== -1) return val;     // unit already baked in
    return val + " " + unit;
  }

  function prettifyKey(key) {
    return String(key).replace(/_/g, " ").replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
  }

  // ── Staged reveal ────────────────────────────────────────────────────────
  // Clicking "Get ESG Data" walks through the pipeline stages one at a time
  // instead of dumping everything on screen at once: show the key players
  // first (names only), then a brief "checking database" beat, then reveal
  // each card's badge/scores.

  function setStage(stageName) {
    Array.prototype.forEach.call(
      stageRail.querySelectorAll(".stage-pill"),
      function (pill) {
        var order = ["players", "db", "scoring"];
        var pillIdx = order.indexOf(pill.dataset.stage);
        var stageIdx = order.indexOf(stageName);
        pill.classList.remove("active", "done");
        if (pillIdx < stageIdx) pill.classList.add("done");
        else if (pillIdx === stageIdx) pill.classList.add("active");
      }
    );
  }

  function resetStages() {
    Array.prototype.forEach.call(
      stageRail.querySelectorAll(".stage-pill"),
      function (pill) { pill.classList.remove("active", "done"); }
    );
  }

  function wait(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function runPipeline(idx) {
    var market = DATA.markets[idx];
    if (!market) return;

    detailPanel.classList.remove("open");
    detailPanel.innerHTML = "";
    marketSummary.classList.remove("visible");
    companyGrid.innerHTML = "";
    runBtn.disabled = true;
    marketSelect.disabled = true;

    setStage("players");
    dbNote.textContent = "Identifying key players in " + market.market + "…";
    renderCompanyGrid(market);

    wait(700)
      .then(function () {
        setStage("db");
        renderDbNote();
        return wait(900);
      })
      .then(function () {
        setStage("scoring");
        dbNote.textContent =
          "Checking database of " + DATA.total_companies_in_db.toLocaleString() +
          " companies — computing ESG scores for " + market.market + "…";
        return wait(900);
      })
      .then(function () {
        revealScores(market);
        renderMarketSummary(market);
        marketSummary.classList.add("visible");
        renderDbNote();
        runBtn.disabled = false;
        marketSelect.disabled = false;
      });
  }

  marketSelect.addEventListener("change", function () {
    resetStages();
    marketSummary.classList.remove("visible");
    companyGrid.innerHTML = "";
    detailPanel.classList.remove("open");
    detailPanel.innerHTML = "";
    renderDbNote();
  });

  runBtn.addEventListener("click", function () {
    runPipeline(Number(marketSelect.value));
  });

  populateMarketSelect();
  renderDbNote();
  if (!DATA.markets.length) {
    companyGrid.innerHTML = '<div class="no-data">No demo data found — run export_demo_data.py first.</div>';
    runBtn.disabled = true;
  }
})();
