(function () {
  "use strict";

  var form = document.getElementById("ask-form");
  var input = document.getElementById("question");
  var kSelect = document.getElementById("k");
  var submit = document.getElementById("submit");
  var notice = document.getElementById("notice");
  var examples = document.getElementById("examples");
  var results = document.getElementById("results");
  var corpus = document.getElementById("corpus");
  var colophon = document.getElementById("colophon");

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var ready = false;
  var busy = false;
  var paperCount = 0;
  var entryCount = 0;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function prettyDate(iso) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso || "");
    if (!m) return "";
    return Number(m[3]) + " " + MONTHS[Number(m[2]) - 1] + " " + m[1];
  }

  function showNotice(text) {
    notice.textContent = text || "";
    notice.hidden = !text;
  }

  function syncControls() {
    var locked = !ready || busy;
    submit.disabled = locked;
    input.disabled = busy;
    examples.querySelectorAll("button").forEach(function (b) { b.disabled = locked; });
  }

  /* ---- status ---------------------------------------------------------- */

  function pollStatus() {
    fetch("/api/status")
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (s.ready) {
          ready = true;
          paperCount = s.papers;
          var latest = prettyDate(s.last_paper_date);
          corpus.textContent = s.papers.toLocaleString("en") + " papers" + (latest ? " · latest " + latest : "");
          colophon.textContent = "Answers written by " + s.model + " from papers found by " +
            s.retrieval_mode.replace("_", " + ") + " search.";
          showNotice(s.llm_configured ? "" :
            "No language-model key is configured on this server, so answers cannot be written. " +
            "The closest papers are still listed.");
          syncControls();
        } else if (s.error) {
          showNotice("The paper index could not be loaded: " + s.error);
        } else {
          showNotice("Reading the papers into the index. This takes about a minute after the server wakes up.");
          setTimeout(pollStatus, 2000);
        }
      })
      .catch(function () {
        showNotice("The server is not answering yet. Retrying.");
        setTimeout(pollStatus, 4000);
      });
  }

  /* ---- answer rendering (text nodes only, never innerHTML) ------------- */

  var INLINE = /\*\*([^*]+)\*\*|`([^`]+)`|\[(\d+(?:\s*,\s*\d+)*)\]/g;

  function appendInline(parent, text, entryId, sourceCount) {
    var last = 0;
    var m;
    INLINE.lastIndex = 0;
    while ((m = INLINE.exec(text)) !== null) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      if (m[1] !== undefined) {
        parent.appendChild(el("strong", "", m[1]));
      } else if (m[2] !== undefined) {
        parent.appendChild(el("code", "", m[2]));
      } else {
        var numbers = m[3].split(",").map(function (n) { return Number(n.trim()); });
        var valid = numbers.every(function (n) { return n >= 1 && n <= sourceCount; });
        if (!valid) {
          parent.appendChild(document.createTextNode(m[0]));
        } else {
          numbers.forEach(function (n) {
            var a = el("a", "cite", "[" + n + "]");
            a.href = "#" + entryId + "-s" + n;
            parent.appendChild(a);
          });
        }
      }
      last = m.index + m[0].length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  var BULLET = /^\s*(?:[-*•]|\d+[.)])\s+/;

  function renderAnswer(container, text, entryId, sourceCount) {
    text.split(/\n\s*\n/).forEach(function (block) {
      var lines = block.split("\n").map(function (l) { return l.trim(); }).filter(Boolean);
      if (!lines.length) return;
      if (lines.every(function (l) { return BULLET.test(l); })) {
        var ordered = /^\s*\d/.test(lines[0]);
        var list = el(ordered ? "ol" : "ul");
        lines.forEach(function (l) {
          var li = el("li");
          appendInline(li, l.replace(BULLET, ""), entryId, sourceCount);
          list.appendChild(li);
        });
        container.appendChild(list);
      } else {
        var p = el("p");
        appendInline(p, lines.join(" ").replace(/^#+\s*/, ""), entryId, sourceCount);
        container.appendChild(p);
      }
    });
  }

  function renderSources(entry, sources, entryId) {
    if (!sources.length) return;
    var section = el("div", "sources");
    section.appendChild(el("h3", "", sources.length === 1 ? "Paper read" : "Papers read"));
    var list = el("ol");
    sources.forEach(function (s) {
      var li = el("li", "source");
      li.id = entryId + "-s" + s.n;
      li.appendChild(el("span", "num", s.n + "."));

      var body = el("div");
      var title = el(s.url ? "a" : "span", "title", s.title);
      if (s.url) { title.href = s.url; title.target = "_blank"; title.rel = "noopener"; }
      body.appendChild(title);

      var meta = el("p", "meta");
      var authors = s.authors.length > 4 ? s.authors.slice(0, 4).join(", ") + ", et al." : s.authors.join(", ");
      var bits = [authors, prettyDate(s.published), s.id ? "arXiv:" + s.id : ""].filter(Boolean);
      meta.appendChild(document.createTextNode(bits.join(" · ")));
      if (s.cited) {
        meta.appendChild(document.createTextNode(" · "));
        meta.appendChild(el("span", "cited", "cited above"));
      }
      body.appendChild(meta);

      if (s.abstract) {
        var details = el("details");
        details.appendChild(el("summary", "", "Abstract"));
        details.appendChild(el("p", "", s.abstract));
        body.appendChild(details);
      }
      li.appendChild(body);
      list.appendChild(li);
    });
    section.appendChild(list);
    entry.appendChild(section);
  }

  /* ---- asking ---------------------------------------------------------- */

  function failureText(status, detail) {
    if (status === 429) return "Too many questions in a short time. Wait a minute and ask again.";
    if (status === 503) return "The paper index is still loading. Try again in a moment.";
    if (status === 422) return "That question is empty or too long (500 characters at most).";
    return detail || "The server could not answer. Try again.";
  }

  function ask(question) {
    question = question.trim();
    if (!question || !ready || busy) return;
    busy = true;
    syncControls();
    examples.hidden = true;

    entryCount += 1;
    var entryId = "q" + entryCount;
    var entry = el("article", "entry");
    entry.appendChild(el("h2", "", question));
    var body = el("div", "answer is-plain");
    body.appendChild(el("p", "pending", "Searching " + paperCount.toLocaleString("en") + " papers and writing an answer"));
    entry.appendChild(body);
    results.insertBefore(entry, results.firstChild);
    entry.scrollIntoView({ behavior: "smooth", block: "start" });

    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, k: Number(kSelect.value) })
    })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (data) {
          return { ok: r.ok, status: r.status, data: data };
        });
      })
      .then(function (res) {
        body.textContent = "";
        if (!res.ok) {
          var detail = typeof res.data.detail === "string" ? res.data.detail : "";
          body.appendChild(el("p", "", failureText(res.status, detail)));
          return;
        }
        var data = res.data;
        var plain = data.degraded || data.refused;
        body.className = plain ? "answer is-plain" : "answer";
        renderAnswer(body, data.answer, entryId, data.sources.length);
        if (data.dropped_count > 0) {
          entry.appendChild(el("p", "remark",
            data.dropped_count === 1
              ? "One sentence was removed because it cited no paper."
              : data.dropped_count + " sentences were removed because they cited no paper."));
        }
        renderSources(entry, data.sources, entryId);
        input.value = "";
      })
      .catch(function () {
        body.textContent = "";
        body.appendChild(el("p", "", "The server could not be reached. Check your connection and try again."));
      })
      .then(function () {
        busy = false;
        syncControls();
      });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    ask(input.value);
  });

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ask(input.value);
    }
  });

  examples.addEventListener("click", function (event) {
    var button = event.target.closest("button");
    if (!button || button.disabled) return;
    input.value = button.textContent;
    ask(button.textContent);
  });

  syncControls();
  pollStatus();
})();
