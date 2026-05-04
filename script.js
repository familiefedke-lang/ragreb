'use strict';

/* ════════════════════════════════════════════════════════
   Markdown shortcode pre-processor
   ════════════════════════════════════════════════════════

   Supported shortcodes (fenced with :::):

   :::box
   Any markdown content for an infobox.
   :::

   :::box2
   Left column markdown content.
   ---split---
   Right column markdown content.
   :::

   :::box3
   First column.
   ---split---
   Second column.
   ---split---
   Third column.
   :::

   :::faq
   Q: Your question here?
   A: Your answer here. Can span multiple lines
      until the next Q: or end of block.

   Q: Another question?
   A: Another answer.
   :::
*/

(function () {

  /* ── helpers ──────────────────────────────────────────── */

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

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ── shortcode pre-processor ──────────────────────────── */

  /**
   * Replace :::type … ::: fenced blocks with HTML placeholders
   * before the Markdown parser sees the text. Each placeholder is
   * a single <div data-md-placeholder="N"> that gets swapped back
   * for the final HTML after marked.parse().
   */
  function processShortcodes(text) {
    var placeholders = [];

    /* Generic fence regex: :::type\ncontent\n::: */
    var fenceRe = /^:::(box3|box2|box|faq)[ \t]*\r?\n([\s\S]*?)\n^:::[ \t]*$/gm;

    text = text.replace(fenceRe, function (_, type, inner) {
      var html = '';

      if (type === 'box') {
        html = renderBox(inner.trim());
      } else if (type === 'box2') {
        html = renderBoxN(inner.trim(), 2);
      } else if (type === 'box3') {
        html = renderBoxN(inner.trim(), 3);
      } else if (type === 'faq') {
        html = renderFaq(inner.trim());
      }

      var idx = placeholders.length;
      placeholders.push(html);
      return '\n\n<div data-md-placeholder="' + idx + '"></div>\n\n';
    });

    return { text: text, placeholders: placeholders };
  }

  function renderBox(inner) {
    return '<div class="md-box">' + marked.parse(inner) + '</div>';
  }

  function renderBoxN(inner, n) {
    var cls = 'md-box' + n;
    var cols = inner.split(/^---split---$/m);
    var html = '<div class="' + cls + '">';
    cols.forEach(function (col) {
      html += '<div class="md-col">' + marked.parse(col.trim()) + '</div>';
    });
    html += '</div>';
    return html;
  }

  function renderFaq(inner) {
    /* Parse Q: / A: pairs */
    var items = [];
    var lines = inner.split('\n');
    var current = null;

    lines.forEach(function (line) {
      var qm = line.match(/^Q:\s*(.*)/);
      var am = line.match(/^A:\s*(.*)/);
      if (qm) {
        current = { q: qm[1], a: [] };
        items.push(current);
      } else if (am && current) {
        current.a.push(am[1]);
      } else if (current && current.a.length) {
        /* continuation of answer */
        current.a.push(line);
      }
    });

    if (!items.length) return '<div class="md-faq"></div>';

    var html = '<div class="md-faq">';
    items.forEach(function (item) {
      var answerMd = item.a.join('\n').trim();
      html += '<details>'
        + '<summary>' + esc(item.q) + '</summary>'
        + '<div class="faq-answer">' + marked.parse(answerMd) + '</div>'
        + '</details>';
    });
    html += '</div>';
    return html;
  }

  /* ── restore placeholders ─────────────────────────────── */

  function restorePlaceholders(htmlStr, placeholders) {
    return htmlStr.replace(
      /<[^>]*data-md-placeholder="(\d+)"[^>]*><\/[^>]+>/g,
      function (_, idx) { return placeholders[parseInt(idx, 10)] || ''; }
    );
  }

  /* ── main ─────────────────────────────────────────────── */

  async function init() {
    try {
      var res = await fetch('content.md');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var raw = await res.text();

      /* ── split preamble / body at first top-level heading ── */
      var splitIdx = raw.search(/\n# /);
      var preamble = splitIdx >= 0 ? raw.slice(0, splitIdx) : '';
      var body     = splitIdx >= 0 ? raw.slice(splitIdx + 1) : raw;

      /* ── extract metadata from preamble ────────────────── */
      var boldRe    = /^\*\*(.+?)\*\*$/;
      var titleParts = [], metaParts = [];
      var inToc = false;

      preamble.split('\n').forEach(function (line) {
        line = line.trim();
        if (!line) return;
        if (line.includes('Inhaltsverzeichnis')) { inToc = true; return; }
        if (inToc || line.startsWith('[')) return;
        var m = line.match(boldRe);
        if (m) { titleParts.push(m[1]); }
        else    { metaParts.push(line); }
      });

      var title    = titleParts.join(' ');
      var subtitle = metaParts[1] || '';
      var category = metaParts[2] || '';
      var date     = metaParts[3] || '';
      var tagLine  = [category, date].filter(Boolean).join(' · ');

      /* ── pre-process markdown ───────────────────────────── */
      /*
       * 1. Strip ** bold markers from headings  (# **Title** → # Title)
       * 2. Shift all heading levels down by one (# → ##, ## → ###)
       *    so H1 is reserved for the page title in <header>.
       * 3. Extract shortcode blocks (:::box, :::box2, :::box3, :::faq)
       */
      var processed = body
        .replace(/^(#{1,6}) \*\*(.+?)\*\*[ \t]*$/gm, '$1 $2')
        .replace(/^(#{1,6} )/gm, function (m) { return '#' + m; });

      var sc = processShortcodes(processed);

      /* ── parse to HTML ──────────────────────────────────── */
      var htmlStr = marked.parse(sc.text);

      /* ── restore shortcode placeholders ────────────────── */
      htmlStr = restorePlaceholders(htmlStr, sc.placeholders);

      /* ── post-process: add IDs, build TOC ──────────────── */
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

      /* ── render article ─────────────────────────────────── */
      var article = document.getElementById('article-body');
      article.innerHTML =
        '<header class="art-header">'
        + (tagLine ? '<p class="art-tag">' + esc(tagLine) + '</p>' : '')
        + '<h1>' + esc(title) + '</h1>'
        + (subtitle ? '<p class="art-subtitle">' + esc(subtitle) + '</p>' : '')
        + '</header>'
        + (toc.length
            ? '<details class="toc-box" open><summary>Inhaltsverzeichnis</summary>'
              + buildInlineToc(toc) + '</details>'
            : '')
        + '<div class="art-body">' + contentHtml + '</div>';

      /* ── render footer nav ──────────────────────────────── */
      document.getElementById('footer-nav').innerHTML =
        '<p class="fnav-label">Navigation</p>' + buildFooterNav(toc);

    } catch (err) {
      var el = document.getElementById('loading') || document.getElementById('article-body');
      if (el) {
        el.textContent = 'Inhalt konnte nicht geladen werden. '
          + 'Bitte stellen Sie sicher, dass die Seite über einen HTTP-Server aufgerufen wird.';
      }
      console.error('Content load error:', err);
    }
  }

  init().catch(function (err) {
    var el = document.getElementById('loading') || document.getElementById('article-body');
    if (el) {
      el.textContent = 'Inhalt konnte nicht geladen werden. '
        + 'Bitte stellen Sie sicher, dass die Seite über einen HTTP-Server aufgerufen wird.';
    }
    console.error('Content load error:', err);
  });

}());
