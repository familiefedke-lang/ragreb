/* script.js – Markdown article loader
 *
 * Usage: add data-md="filename.md" to the <article id="article-body">
 * element to specify which markdown file to load.
 * Defaults to "content.md".
 *
 * Requires: marked.js (loaded before this script via CDN or locally)
 */

(async function () {
  'use strict';

  /* ── Helpers ───────────────────────────────────────── */
  function esc(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function slugify(text) {
    return text
      .toLowerCase()
      .replace(/[|]/g, '')
      .replace(/\s+/g, '-')
      .replace(/[^\w\-]/g, '')
      .replace(/-{2,}/g, '-')
      .replace(/^-|-$/g, '');
  }

  function buildInlineToc(items) {
    if (!items.length) return '';
    let h = '<ul class="toc-list">';
    items.forEach(function (it) {
      h += '<li class="' + (it.level === 2 ? 'toc-l2' : 'toc-l3') + '">'
        + '<a href="#' + it.id + '">' + esc(it.text) + '</a></li>';
    });
    return h + '</ul>';
  }

  function buildFooterNav(items) {
    if (!items.length) return '';
    let h = '<ul class="fnav-list">';
    items.forEach(function (it) {
      var cls = it.level === 2 ? 'fnav-ch' : 'fnav-sec';
      h += '<li class="' + cls + '"><a href="#' + it.id + '">' + esc(it.text) + '</a></li>';
    });
    return h + '</ul>';
  }

  /* ── Preamble parser ────────────────────────────────
   * Handles both plain preamble (content.md) and
   * markdown-table preamble (vektorartikel.md).        */
  function parsePreamble(raw) {
    var boldRe = /^\*\*(.+?)\*\*$/;
    var titleParts = [], metaParts = [];
    var inToc = false;
    var hasTableFormat = /^\|/.test(raw.trim());

    raw.split('\n').forEach(function (line) {
      line = line.trim();
      if (!line) return;

      /* Table-format: strip leading/trailing pipes */
      if (hasTableFormat && line.startsWith('|')) {
        line = line.replace(/^\|\s*/, '').replace(/\s*\|$/, '').trim();
      }
      if (!line) return;

      /* Skip markdown table separator rows (e.g. "---" or "| --- |") */
      if (/^[-\s]+$/.test(line)) return;

      if (line.includes('Inhaltsverzeichnis')) { inToc = true; return; }
      if (inToc || line.startsWith('[')) return;

      var m = line.match(boldRe);
      if (m) { titleParts.push(m[1]); }
      else    { metaParts.push(line); }
    });

    var title = titleParts.join(' ');
    var subtitle, tagLine;

    if (hasTableFormat) {
      /* vektorartikel.md-style: [0]=subtitle, [1]=tagLine */
      subtitle = metaParts[0] || '';
      tagLine  = (metaParts[1] || '').replace(/\\\|/g, '|');
    } else {
      /* content.md-style: [0]=unused, [1]=subtitle, [2]=category, [3]=date */
      subtitle = metaParts[1] || '';
      var category = metaParts[2] || '';
      var date     = metaParts[3] || '';
      tagLine = [category, date].filter(Boolean).join(' · ');
    }

    return { title: title, subtitle: subtitle, tagLine: tagLine };
  }

  /* ── Main ───────────────────────────────────────────── */
  var article = document.getElementById('article-body');
  if (!article) return;

  var mdFile = article.dataset.md || 'content.md';

  try {
    var res = await fetch(mdFile);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    var raw = await res.text();

    /* Split at first top-level heading */
    var splitIdx = raw.search(/\n# /);
    var preamble = splitIdx >= 0 ? raw.slice(0, splitIdx) : '';
    var body     = splitIdx >= 0 ? raw.slice(splitIdx + 1) : raw;

    /* Extract metadata */
    var meta = parsePreamble(preamble);

    /* Pre-process markdown body:
     * 1. Strip ** bold markers from headings  (# **Title** → # Title)
     * 2. Shift all heading levels down by one (# → ##, ## → ###)
     *    so H1 is reserved for the page title in <header>.
     */
    var processed = body
      .replace(/^(#{1,6}) \*\*(.+?)\*\*[ \t]*$/gm, '$1 $2')
      .replace(/^(#{1,6} )/gm, function (m) { return '#' + m; });

    /* Parse markdown to HTML (requires marked.js) */
    var htmlStr = (typeof marked !== 'undefined')
      ? marked.parse(processed)
      : '<pre>' + esc(processed) + '</pre>';

    /* Post-process: add IDs to headings, build TOC */
    var domParser = new DOMParser();
    var doc = domParser.parseFromString(htmlStr, 'text/html');
    var toc = [], seen = new Set();

    doc.querySelectorAll('h2, h3').forEach(function (el) {
      var id = slugify(el.textContent);
      if (seen.has(id)) {
        var n = 2;
        while (seen.has(id + '-' + n)) n++;
        id = id + '-' + n;
      }
      seen.add(id);
      el.id = id;
      toc.push({ level: el.tagName === 'H2' ? 2 : 3, text: el.textContent.trim(), id: id });
    });

    var contentHtml = doc.body.innerHTML;

    /* Update page title */
    if (meta.title) document.title = meta.title;

    /* Render article */
    article.innerHTML =
      '<header class="art-header">'
      + (meta.tagLine ? '<p class="art-tag">' + esc(meta.tagLine) + '</p>' : '')
      + '<h1>' + esc(meta.title || '') + '</h1>'
      + (meta.subtitle ? '<p class="art-subtitle">' + esc(meta.subtitle) + '</p>' : '')
      + '</header>'
      + (toc.length
          ? '<details class="toc-box" open><summary>Inhaltsverzeichnis</summary>'
            + buildInlineToc(toc) + '</details>'
          : '')
      + '<div class="art-body">' + contentHtml + '</div>';

    /* Render footer nav */
    var footerNav = document.getElementById('footer-nav');
    if (footerNav && toc.length) {
      footerNav.innerHTML = '<p class="fnav-label">Navigation</p>' + buildFooterNav(toc);
    }

  } catch (err) {
    var el = document.getElementById('loading') || article;
    if (el) {
      el.textContent = 'Inhalt konnte nicht geladen werden. '
        + 'Bitte stellen Sie sicher, dass die Seite über einen HTTP-Server aufgerufen wird.';
    }
    console.error('Content load error:', err);
  }

}());

