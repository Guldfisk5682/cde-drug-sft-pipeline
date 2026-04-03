// ==UserScript==
// @name         CDE Leaflet Helper
// @namespace    https://github.com/Guldfisk5682/cde-drug-sft-pipeline
// @version      0.2.0
// @description  Semi-automatic capture helper for CDE drug detail pages and leaflet attachments.
// @match        https://www.cde.org.cn/*
// @grant        GM_setClipboard
// @run-at       document-start
// ==/UserScript==

(function () {
  "use strict";

  const STORAGE_KEY = "cde-drug-sft-pipeline.userscript.session.v1";
  const PANEL_ID = "cde-leaflet-helper-panel";
  const AUTO_CAPTURE_HASH = "#cde-helper-auto-detail";
  const LIST_API_PATTERN = /\/hymlj\/getList\b/i;
  const DOWNLOAD_ID_PATTERN = /downloadFile\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)/i;
  const runtimeState = {
    latestListApi: null,
    listApiHistory: [],
    autoCaptureStarted: false,
  };

  function nowIso() {
    return new Date().toISOString();
  }

  function textOf(node) {
    return (node?.textContent || "").replace(/\s+/g, " ").trim();
  }

  function absoluteUrl(href) {
    if (!String(href || "").trim()) {
      return "";
    }
    try {
      return new URL(href, window.location.href).toString();
    } catch (_) {
      return "";
    }
  }

  function loadSession() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return {
          created_at: nowIso(),
          updated_at: nowIso(),
          list_pages: [],
          detail_pages: [],
          api_captures: [],
          crawl_state: null,
        };
      }
      const parsed = JSON.parse(raw);
      parsed.list_pages = Array.isArray(parsed.list_pages) ? parsed.list_pages : [];
      parsed.detail_pages = Array.isArray(parsed.detail_pages) ? parsed.detail_pages : [];
      parsed.api_captures = Array.isArray(parsed.api_captures) ? parsed.api_captures : [];
      parsed.crawl_state = parsed.crawl_state || null;
      return parsed;
    } catch (_) {
      return {
        created_at: nowIso(),
        updated_at: nowIso(),
        list_pages: [],
        detail_pages: [],
        api_captures: [],
        crawl_state: null,
      };
    }
  }

  function saveSession(session) {
    session.updated_at = nowIso();
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session, null, 2));
  }

  function uniqueBy(items, keyFn) {
    const seen = new Set();
    return items.filter((item) => {
      const key = keyFn(item);
      if (!key || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  function dedupeByUrl(items) {
    return uniqueBy(items.filter(Boolean), (item) => item.url || item.detail_url || item);
  }

  function safeJsonParse(rawText) {
    try {
      return JSON.parse(rawText);
    } catch (_) {
      return null;
    }
  }

  function rememberListApiCapture(entry) {
    runtimeState.latestListApi = entry;
    runtimeState.listApiHistory.push(entry);
    if (runtimeState.listApiHistory.length > 10) {
      runtimeState.listApiHistory = runtimeState.listApiHistory.slice(-10);
    }
  }

  function normalizeListApiPayload(payload, requestUrl) {
    const records = payload?.data?.records;
    if (!Array.isArray(records)) {
      return null;
    }

    const normalizedRecords = records.map((record, index) => {
      const idCode = String(record.idCode || "").trim();
      return {
        index: index + 1,
        id_code: idCode,
        drug_name: String(record.ypmc || "").trim(),
        dosage_form: String(record.jx || "").trim(),
        spec: String(record.gg || "").trim(),
        reference_drug: String(record.cbzj || "").trim(),
        approval_number: String(record.pzwh || "").trim(),
        approval_date: String(record.scpzrq || "").trim(),
        manufacturer: String(record.sccs || "").trim(),
        market_authorization_holder: String(record.ssxkzcyr || "").trim(),
        detail_url: idCode ? absoluteUrl(`/hymlj/detailPage/${idCode}`) : "",
      };
    });

    return {
      captured_at: nowIso(),
      request_url: requestUrl,
      record_count: normalizedRecords.length,
      records: normalizedRecords,
    };
  }

  function installNetworkHooks() {
    if (window.__CDE_LEAFLET_HELPER_HOOKED__) {
      return;
    }
    window.__CDE_LEAFLET_HELPER_HOOKED__ = true;

    const originalFetch = window.fetch;
    if (typeof originalFetch === "function") {
      window.fetch = async function (...args) {
        const response = await originalFetch.apply(this, args);
        try {
          const requestUrl = typeof args[0] === "string" ? args[0] : args[0]?.url || "";
          const absoluteRequestUrl = absoluteUrl(requestUrl);
          if (LIST_API_PATTERN.test(absoluteRequestUrl)) {
            const clone = response.clone();
            const text = await clone.text();
            const payload = safeJsonParse(text);
            const normalized = normalizeListApiPayload(payload, absoluteRequestUrl);
            if (normalized) {
              rememberListApiCapture(normalized);
            }
          }
        } catch (_) {
          // Ignore instrumentation failures and let the page continue normally.
        }
        return response;
      };
    }

    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;

    XMLHttpRequest.prototype.open = function (method, url, ...rest) {
      this.__cdeHelperUrl = absoluteUrl(String(url || ""));
      return originalOpen.call(this, method, url, ...rest);
    };

    XMLHttpRequest.prototype.send = function (...args) {
      this.addEventListener("load", function () {
        try {
          const requestUrl = this.__cdeHelperUrl || "";
          if (!LIST_API_PATTERN.test(requestUrl)) {
            return;
          }
          const payload = safeJsonParse(this.responseText);
          const normalized = normalizeListApiPayload(payload, requestUrl);
          if (normalized) {
            rememberListApiCapture(normalized);
          }
        } catch (_) {
          // Ignore instrumentation failures and let the page continue normally.
        }
      });
      return originalSend.apply(this, args);
    };
  }

  function looksLikeDetailUrl(urlText) {
    return /\/detailpage(\/|$)/i.test(urlText) || /postmarketpage/i.test(urlText);
  }

  function looksLikeListUrl(urlText) {
    return /\/listpage(\/|$)/i.test(urlText);
  }

  function hasDetailAnchors() {
    return collectAnchors().some((item) => {
      return looksLikeDetailUrl(item.href);
    });
  }

  function pageKind() {
    const href = window.location.href;
    const path = window.location.pathname || "";
    const search = window.location.search || "";
    const hash = window.location.hash || "";
    const routeText = `${href} ${path} ${search} ${hash}`;

    if (looksLikeDetailUrl(routeText)) {
      return "detail";
    }
    if (looksLikeListUrl(routeText)) {
      return "list";
    }

    // DOM fallback for dynamic routes where pathname is rewritten or hidden in hash/query.
    if (hasDetailAnchors()) {
      return "list";
    }

    return "unknown";
  }

  function collectAnchors() {
    return Array.from(document.querySelectorAll("a[href]"))
      .map((anchor) => ({
        text: textOf(anchor),
        href: absoluteUrl(anchor.getAttribute("href") || ""),
      }))
      .filter((item) => item.href);
  }

  function collectButtons() {
    return Array.from(document.querySelectorAll("button, [role='button']"))
      .map((button) => ({
        text: textOf(button),
        onclick: button.getAttribute("onclick") || "",
      }))
      .filter((item) => item.text || item.onclick);
  }

  function tableRows() {
    const rows = Array.from(document.querySelectorAll("tr"));
    return rows.map((row) => {
      const cells = Array.from(row.querySelectorAll("th, td"))
        .map((cell) => textOf(cell))
        .filter(Boolean);
      const links = Array.from(row.querySelectorAll("a[href]")).map((anchor) => ({
        text: textOf(anchor),
        href: absoluteUrl(anchor.getAttribute("href") || ""),
      }));
      return {
        text: cells.join(" | "),
        cells,
        links,
      };
    }).filter((row) => row.text);
  }

  function parseDownloadCall(onclickValue) {
    const match = String(onclickValue || "").match(DOWNLOAD_ID_PATTERN);
    if (!match) {
      return null;
    }

    const attachmentId = match[1].trim();
    const fileName = match[2].trim();

    return {
      attachment_id: attachmentId,
      file_name: fileName,
      download_endpoint: absoluteUrl(`/hymlj/download/sms/${attachmentId}`),
      raw_onclick: String(onclickValue || ""),
    };
  }

  function collectLayuiRecordsFromDom() {
    const rows = Array.from(document.querySelectorAll("tr[data-index]"));
    const records = rows.map((row, index) => {
      const cells = Array.from(row.querySelectorAll("td[data-field]"));
      const mapped = {};
      for (const cell of cells) {
        const fieldName = cell.getAttribute("data-field") || "";
        mapped[fieldName] = textOf(cell);
      }

      const detailAnchor = row.querySelector("td[data-field='ypmc'] a[href]");
      const detailUrl = absoluteUrl(detailAnchor?.getAttribute("href") || "");
      return {
        index: index + 1,
        id_code: detailUrl.split("/").pop() || "",
        drug_name: mapped.ypmc || textOf(detailAnchor),
        dosage_form: mapped.jx || "",
        spec: mapped.gg || "",
        reference_drug: mapped.cbzj || "",
        approval_number: mapped.pzwh || "",
        approval_date: mapped.scpzrq || "",
        manufacturer: mapped.sccs || "",
        market_authorization_holder: mapped.ssxkzcyr || "",
        detail_url: detailUrl,
      };
    });

    return records.filter((record) => record.drug_name || record.detail_url);
  }

  function collectPaginationState() {
    const currentNode = document.querySelector(".layui-laypage-curr em:last-child");
    const nextNode = document.querySelector(".layui-laypage-next[data-page]");
    return {
      current_page: textOf(currentNode),
      next_page: nextNode?.getAttribute("data-page") || "",
      has_next_page: Boolean(nextNode),
    };
  }

  function detectLabelValueBlocks() {
    const blocks = [];
    const candidates = Array.from(document.querySelectorAll("li, p, div, td, th"));
    for (const node of candidates) {
      const value = textOf(node);
      if (!value || value.length > 200) {
        continue;
      }
      if (/[:：]/.test(value) && value.length >= 4) {
        blocks.push(value);
      }
    }
    return uniqueBy(blocks, (item) => item);
  }

  function collectListPage() {
    const detailAnchors = collectAnchors().filter((item) => {
      return looksLikeDetailUrl(item.href);
    });
    const apiCapture = runtimeState.latestListApi;
    const domRecords = collectLayuiRecordsFromDom();
    const records = apiCapture?.records?.length ? apiCapture.records : domRecords;
    const pagination = collectPaginationState();

    return {
      captured_at: nowIso(),
      url: window.location.href,
      title: document.title,
      page_kind: "list",
      source: apiCapture?.records?.length ? "list_api" : "dom",
      detail_anchor_count: detailAnchors.length,
      api_capture: apiCapture,
      pagination,
      candidate_records: records,
      raw_detail_anchors: detailAnchors,
      raw_buttons: collectButtons(),
    };
  }

  function collectDetailPage() {
    const anchors = collectAnchors();
    const canonicalUrl = absoluteUrl(window.location.pathname);
    const attachmentLinks = Array.from(document.querySelectorAll("a.download, a[onclick*='downloadFile']"))
      .map((anchor) => {
        const downloadMeta = parseDownloadCall(anchor.getAttribute("onclick") || "");
        const rawHref = String(anchor.getAttribute("href") || "").trim();
        const resolvedHref = absoluteUrl(rawHref);
        return {
          text: textOf(anchor),
          href:
            resolvedHref && resolvedHref !== canonicalUrl
              ? resolvedHref
              : (downloadMeta?.download_endpoint || ""),
          ...downloadMeta,
        };
      })
      .filter((item) => item.attachment_id || item.href);

    const headings = Array.from(document.querySelectorAll("h1, h2, h3, h4, .title, .name"))
      .map((node) => textOf(node))
      .filter(Boolean);

    const pageText = textOf(document.body).slice(0, 20000);
    const rxHints = [
      "请仔细阅读说明书并在医师指导下使用",
      "处方药",
      "非处方药",
      "otc",
      "rx",
    ].filter((hint) => pageText.toLowerCase().includes(hint.toLowerCase()));

    return {
      captured_at: nowIso(),
      url: canonicalUrl,
      full_url: window.location.href,
      canonical_url: canonicalUrl,
      title: document.title,
      page_kind: "detail",
      headings,
      label_value_hints: detectLabelValueBlocks().slice(0, 200),
      attachment_links: uniqueBy(attachmentLinks, (item) => item.attachment_id || item.href),
      rx_hints: rxHints,
      anchor_count: anchors.length,
      page_excerpt: pageText.slice(0, 3000),
    };
  }

  function mergeIntoSession(entry) {
    const session = loadSession();
    if (entry.page_kind === "list") {
      session.list_pages = session.list_pages.filter((item) => item.url !== entry.url);
      session.list_pages.push(entry);
    } else if (entry.page_kind === "detail") {
      const entryKey = entry.canonical_url || entry.url;
      session.detail_pages = session.detail_pages.filter((item) => {
        const itemKey = item.canonical_url || item.url;
        return itemKey !== entryKey;
      });
      session.detail_pages.push(entry);
    }
    if (runtimeState.latestListApi) {
      session.api_captures = uniqueBy(
        [...(session.api_captures || []), runtimeState.latestListApi],
        (item) => `${item.request_url}_${item.captured_at}`
      ).slice(-10);
    }
    saveSession(session);
    return session;
  }

  function saveCrawlState(crawlState) {
    const session = loadSession();
    session.crawl_state = crawlState;
    saveSession(session);
    return session;
  }

  function normalizeDetailQueueItem(record, sourcePage) {
    const detailUrl = String(record.detail_url || "").trim();
    if (!detailUrl) {
      return null;
    }
    return {
      url: absoluteUrl(detailUrl),
      id_code: String(record.id_code || "").trim(),
      drug_name: String(record.drug_name || "").trim(),
      dosage_form: String(record.dosage_form || "").trim(),
      spec: String(record.spec || "").trim(),
      approval_number: String(record.approval_number || "").trim(),
      source_page_url: sourcePage.url,
      source_page_number: sourcePage.pagination?.current_page || "",
      queued_at: nowIso(),
    };
  }

  function queueCurrentListDetails() {
    const listEntry = collectListPage();
    const session = mergeIntoSession(listEntry);
    const existingCrawlState = session.crawl_state || null;
    const shouldResetCompleted =
      !existingCrawlState?.active &&
      (((existingCrawlState?.pending_details) || []).length === 0);
    const existingPending = new Set(
      ((existingCrawlState?.pending_details) || []).map((item) => item.url).filter(Boolean)
    );
    const existingCompleted = new Set(
      (shouldResetCompleted ? [] : ((existingCrawlState?.completed_details) || []))
        .map((item) => item.url)
        .filter(Boolean)
    );

    const newQueue = dedupeByUrl(
      listEntry.candidate_records
        .map((record) => normalizeDetailQueueItem(record, listEntry))
        .filter((item) => {
          if (!item?.url) {
            return false;
          }
          return !existingPending.has(item.url) && !existingCompleted.has(item.url);
        })
    );

    const crawlState = {
      active: false,
      created_at: existingCrawlState?.created_at || nowIso(),
      updated_at: nowIso(),
      source_list_url: listEntry.url,
      source_page_number: listEntry.pagination?.current_page || "",
      pending_details: [
        ...((existingCrawlState?.pending_details) || []),
        ...newQueue,
      ],
      completed_details: shouldResetCompleted ? [] : ((existingCrawlState?.completed_details) || []),
      last_detail_url: existingCrawlState?.last_detail_url || "",
    };

    saveCrawlState(crawlState);
    const totalQueued = crawlState.pending_details.length;
    renderPanel(loadSession(), `Queued ${newQueue.length} detail pages (${totalQueued} pending)`);
  }

  function nextPendingDetail(session = loadSession()) {
    const pending = session.crawl_state?.pending_details || [];
    return pending.length ? pending[0] : null;
  }

  function navigateToNextQueuedDetail() {
    let session = loadSession();
    let nextItem = nextPendingDetail(session);
    if (!nextItem && pageKind() === "list") {
      queueCurrentListDetails();
      session = loadSession();
      nextItem = nextPendingDetail(session);
    }
    if (!nextItem?.url) {
      renderPanel(session, "No queued detail pages");
      return;
    }

    const crawlState = {
      ...(session.crawl_state || {}),
      active: true,
      updated_at: nowIso(),
      last_detail_url: nextItem.url,
    };
    saveCrawlState(crawlState);
    window.location.href = `${nextItem.url}${AUTO_CAPTURE_HASH}`;
  }

  function finishCurrentDetailAndContinue(detailEntry) {
    const session = loadSession();
    const crawlState = session.crawl_state;
    if (!crawlState) {
      return;
    }

    const currentUrl = absoluteUrl(window.location.pathname);
    const pending = Array.isArray(crawlState.pending_details) ? crawlState.pending_details : [];
    const completed = Array.isArray(crawlState.completed_details) ? crawlState.completed_details : [];
    const currentItem = pending.find((item) => item.url === currentUrl || item.url === detailEntry.url || item.url === detailEntry.canonical_url);
    const remaining = pending.filter((item) => item.url !== currentUrl && item.url !== detailEntry.url && item.url !== detailEntry.canonical_url);
    const nextCompleted = dedupeByUrl([
      ...completed,
      {
        ...(currentItem || {}),
        url: currentUrl,
        captured_at: detailEntry.captured_at,
        attachment_count: detailEntry.attachment_links?.length || 0,
      },
    ]);

    const nextState = {
      ...crawlState,
      active: remaining.length > 0,
      updated_at: nowIso(),
      pending_details: remaining,
      completed_details: nextCompleted,
      last_detail_url: currentUrl,
    };
    saveCrawlState(nextState);

    if (remaining.length > 0) {
      const nextUrl = remaining[0].url;
      window.setTimeout(() => {
        window.location.href = `${nextUrl}${AUTO_CAPTURE_HASH}`;
      }, 900);
      return;
    }

    const returnUrl = crawlState.source_list_url || absoluteUrl("/hymlj/listpage/9cd8db3b7530c6fa0c86485e563f93c7");
    window.setTimeout(() => {
      window.location.href = returnUrl;
    }, 900);
  }

  function maybeAutoCaptureDetailPage() {
    if (runtimeState.autoCaptureStarted) {
      return;
    }
    const session = loadSession();
    const crawlState = session.crawl_state;
    const currentUrl = absoluteUrl(window.location.pathname);
    const shouldAutoCapture =
      pageKind() === "detail" &&
      Boolean(crawlState?.active) &&
      (window.location.hash === AUTO_CAPTURE_HASH ||
        (crawlState?.pending_details || []).some((item) => item.url === currentUrl));

    if (!shouldAutoCapture) {
      return;
    }

    runtimeState.autoCaptureStarted = true;
    const attemptCapture = (remainingAttempts) => {
      const entry = collectDetailPage();
      const attachmentCount = entry.attachment_links?.length || 0;
      if (attachmentCount > 0 || remainingAttempts <= 0) {
        mergeIntoSession(entry);
        renderPanel(loadSession(), `Captured detail (${attachmentCount} attachments)`);
        finishCurrentDetailAndContinue(entry);
        return;
      }
      window.setTimeout(() => attemptCapture(remainingAttempts - 1), 700);
    };

    window.setTimeout(() => attemptCapture(8), 1200);
  }

  function exportSession() {
    const session = loadSession();
    const blob = new Blob([JSON.stringify(session, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `cde_capture_session_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function copySummary() {
    const session = loadSession();
    const summary = {
      updated_at: session.updated_at,
      list_pages: session.list_pages.length,
      detail_pages: session.detail_pages.length,
      api_captures: (session.api_captures || []).length,
      attachment_links: session.detail_pages.reduce((count, page) => {
        return count + (page.attachment_links?.length || 0);
      }, 0),
    };
    const text = JSON.stringify(summary, null, 2);
    if (typeof GM_setClipboard === "function") {
      GM_setClipboard(text);
    } else if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
  }

  function clearSession() {
    window.localStorage.removeItem(STORAGE_KEY);
    runtimeState.latestListApi = null;
    runtimeState.listApiHistory = [];
    runtimeState.autoCaptureStarted = false;
    renderPanel();
  }

  function openNextPage() {
    const nextNode = document.querySelector(".layui-laypage-next[data-page]");
    if (nextNode instanceof HTMLElement) {
      nextNode.click();
    }
  }

  function captureCurrentPage() {
    const kind = pageKind();
    let entry;
    if (kind === "list") {
      entry = collectListPage();
    } else if (kind === "detail") {
      entry = collectDetailPage();
    } else {
      entry = {
        captured_at: nowIso(),
        url: window.location.href,
        title: document.title,
        page_kind: "unknown",
        anchors: collectAnchors().slice(0, 100),
        label_value_hints: detectLabelValueBlocks().slice(0, 100),
      };
    }
    const session = mergeIntoSession(entry);
    renderPanel(session, `Captured ${kind} page`);
  }

  function styleButton(button) {
    button.style.display = "block";
    button.style.width = "100%";
    button.style.margin = "6px 0";
    button.style.padding = "8px 10px";
    button.style.border = "1px solid #cbd5e1";
    button.style.borderRadius = "8px";
    button.style.background = "#f8fafc";
    button.style.cursor = "pointer";
    button.style.fontSize = "12px";
    button.style.textAlign = "left";
  }

  function renderPanel(session = loadSession(), message = "") {
    let panel = document.getElementById(PANEL_ID);
    if (!panel) {
      panel = document.createElement("div");
      panel.id = PANEL_ID;
      document.body.appendChild(panel);
    }

    panel.innerHTML = "";
    panel.style.position = "fixed";
    panel.style.top = "16px";
    panel.style.right = "16px";
    panel.style.zIndex = "2147483647";
    panel.style.width = "280px";
    panel.style.padding = "12px";
    panel.style.border = "1px solid #cbd5e1";
    panel.style.borderRadius = "12px";
    panel.style.background = "rgba(255,255,255,0.97)";
    panel.style.boxShadow = "0 10px 30px rgba(15,23,42,0.15)";
    panel.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
    panel.style.fontSize = "12px";
    panel.style.color = "#0f172a";

    const title = document.createElement("div");
    title.textContent = "CDE Leaflet Helper";
    title.style.fontWeight = "700";
    title.style.marginBottom = "8px";
    panel.appendChild(title);

    const status = document.createElement("div");
    status.textContent = `Page: ${pageKind()} | list ${session.list_pages.length} | detail ${session.detail_pages.length}`;
    status.style.marginBottom = "8px";
    panel.appendChild(status);

    if (session.crawl_state) {
      const crawlStatus = document.createElement("div");
      const pendingCount = session.crawl_state.pending_details?.length || 0;
      const completedCount = session.crawl_state.completed_details?.length || 0;
      crawlStatus.textContent = `Queue: pending ${pendingCount} | done ${completedCount}`;
      crawlStatus.style.marginBottom = "8px";
      crawlStatus.style.color = "#7c3aed";
      panel.appendChild(crawlStatus);
    }

    if (runtimeState.latestListApi?.record_count) {
      const apiStatus = document.createElement("div");
      apiStatus.textContent = `API records: ${runtimeState.latestListApi.record_count}`;
      apiStatus.style.marginBottom = "8px";
      apiStatus.style.color = "#1d4ed8";
      panel.appendChild(apiStatus);
    }

    if (message) {
      const flash = document.createElement("div");
      flash.textContent = message;
      flash.style.marginBottom = "8px";
      flash.style.color = "#166534";
      panel.appendChild(flash);
    }

    const captureBtn = document.createElement("button");
    captureBtn.textContent = "Capture Current Page";
    captureBtn.addEventListener("click", captureCurrentPage);
    styleButton(captureBtn);
    panel.appendChild(captureBtn);

    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export Session JSON";
    exportBtn.addEventListener("click", exportSession);
    styleButton(exportBtn);
    panel.appendChild(exportBtn);

    if (pageKind() === "list") {
      const queueBtn = document.createElement("button");
      queueBtn.textContent = "Queue Current Page Details";
      queueBtn.addEventListener("click", queueCurrentListDetails);
      styleButton(queueBtn);
      panel.appendChild(queueBtn);

      const crawlBtn = document.createElement("button");
      crawlBtn.textContent = "Start Auto Detail Crawl";
      crawlBtn.addEventListener("click", navigateToNextQueuedDetail);
      styleButton(crawlBtn);
      panel.appendChild(crawlBtn);

      const nextBtn = document.createElement("button");
      nextBtn.textContent = "Open Next Page";
      nextBtn.addEventListener("click", openNextPage);
      styleButton(nextBtn);
      panel.appendChild(nextBtn);
    }

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy Capture Summary";
    copyBtn.addEventListener("click", copySummary);
    styleButton(copyBtn);
    panel.appendChild(copyBtn);

    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear Session";
    clearBtn.addEventListener("click", clearSession);
    styleButton(clearBtn);
    panel.appendChild(clearBtn);
  }

  installNetworkHooks();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      renderPanel();
      maybeAutoCaptureDetailPage();
    });
  } else {
    renderPanel();
    maybeAutoCaptureDetailPage();
  }
})();
