(() => {
  "use strict";

  const data = window.LOWRANKARENA_DATA;

  if (!data) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      '<p class="data-error">Leaderboard data could not be loaded.</p>'
    );
    return;
  }

  const state = {
    model: "llama-31-8b",
    keep: "all",
    view: "summary",
    comparison: "mcq",
    profile: "prefill"
  };

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const setPressed = (buttons, activeButton) => {
    buttons.forEach((button) => {
      const active = button === activeButton;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
      if (button.getAttribute("role") === "tab") {
        button.setAttribute("aria-selected", String(active));
        button.tabIndex = active ? 0 : -1;
      }
    });
  };

  const buildModelTabs = () => {
    const container = document.querySelector("#model-tabs");

    Object.entries(data.accuracy).forEach(([key, model]) => {
      const button = document.createElement("button");
      const active = key === state.model;
      button.type = "button";
      button.textContent = model.label;
      button.dataset.model = key;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(active));
      button.setAttribute("aria-pressed", String(active));
      button.tabIndex = active ? 0 : -1;
      button.classList.toggle("is-active", active);
      button.addEventListener("click", () => {
        state.model = key;
        setPressed([...container.querySelectorAll("button")], button);
        renderAccuracy();
      });
      container.append(button);
    });
  };

  const contiguousGroups = (columns) => {
    const groups = [];

    columns.forEach((column) => {
      const label = column.group || "";
      const previous = groups.at(-1);
      if (previous && previous.label === label) {
        previous.span += 1;
      } else {
        groups.push({ label, span: 1 });
      }
    });

    return groups;
  };

  const accuracyBestValues = (rows, columns) => {
    const best = new Map();

    ["80%", "60%"].forEach((keep) => {
      const candidates = rows.filter((row) => row.keep === keep);
      columns.forEach((column) => {
        if (!column.direction || candidates.length === 0) return;
        const values = candidates.map((row) => row[column.key]);
        best.set(
          `${keep}:${column.key}`,
          column.direction === "min" ? Math.min(...values) : Math.max(...values)
        );
      });
    });

    return best;
  };

  const renderAccuracy = () => {
    const table = document.querySelector("#accuracy-table");
    const model = data.accuracy[state.model];
    const columns = data.accuracyColumns.filter(
      (column) => column.always || state.view === "full" || column.summary
    );
    const numericColumns = columns.filter((column) => !column.always);
    const rows = model.rows.filter(
      (row) => state.keep === "all" || row.keep === "100%" || row.keep === state.keep
    );
    const best = accuracyBestValues(model.rows, numericColumns);
    const groups = contiguousGroups(numericColumns);

    const groupHead = groups
      .map(
        (group) =>
          `<th scope="colgroup" colspan="${group.span}">${escapeHtml(group.label)}</th>`
      )
      .join("");
    const columnHead = numericColumns
      .map((column) => `<th scope="col">${escapeHtml(column.label)}</th>`)
      .join("");

    let previousKeep = null;
    const body = rows
      .map((row) => {
        const dense = row.keep === "100%";
        const groupStart = previousKeep !== null && previousKeep !== row.keep;
        previousKeep = row.keep;
        const classes = [dense ? "dense-row" : "", groupStart ? "group-start" : ""]
          .filter(Boolean)
          .join(" ");
        const cells = columns
          .map((column) => {
            const value = row[column.key];
            const formatted = column.type === "text"
              ? escapeHtml(value)
              : Number(value).toFixed(column.decimals);
            const isBest =
              !dense &&
              column.direction &&
              value === best.get(`${row.keep}:${column.key}`);
            return `<td${isBest ? ' class="best-cell"' : ""}>${formatted}</td>`;
          })
          .join("");
        return `<tr${classes ? ` class="${classes}"` : ""}>${cells}</tr>`;
      })
      .join("");

    table.classList.toggle("is-full", state.view === "full");
    table.innerHTML = `
      <caption>${escapeHtml(model.label)} standardized quality leaderboard</caption>
      <thead>
        <tr class="group-head">
          <th scope="col" rowspan="2">Keep</th>
          <th scope="col" rowspan="2">Method</th>
          ${groupHead}
        </tr>
        <tr>${columnHead}</tr>
      </thead>
      <tbody>${body}</tbody>
    `;

    const compressedCount = rows.filter((row) => row.keep !== "100%").length;
    const keepLabel = state.keep === "all" ? "80% and 60% keep" : `${state.keep} keep`;
    document.querySelector("#accuracy-status").textContent =
      `${model.label} · ${keepLabel} · ${compressedCount} compressed rows`;
  };

  const renderComparison = () => {
    const table = document.querySelector("#comparison-table");
    const comparison = data.compressionComparison;
    const metric = comparison.metrics[state.comparison];
    const candidates = metric.rows.filter((row) => !row.dense);
    const best = comparison.ratios.map((_, index) => {
      const values = candidates.map((row) => row.values[index]);
      return metric.direction === "min" ? Math.min(...values) : Math.max(...values);
    });

    let previousCategory = null;
    const body = metric.rows
      .map((row) => {
        const groupStart = previousCategory !== null && previousCategory !== row.category;
        previousCategory = row.category;
        const rowClasses = [row.dense ? "dense-row" : "", groupStart ? "group-start" : ""]
          .filter(Boolean)
          .join(" ");
        const values = row.values
          .map((value, index) => {
            const isBest = !row.dense && value === best[index];
            return `<td${isBest ? ' class="best-cell"' : ""}>${value.toFixed(metric.decimals)}</td>`;
          })
          .join("");
        return `
          <tr${rowClasses ? ` class="${rowClasses}"` : ""}>
            <td class="category-cell">${escapeHtml(row.category)}</td>
            <td>${escapeHtml(row.method)}</td>
            <td>${escapeHtml(row.training)}</td>
            ${values}
          </tr>
        `;
      })
      .join("");

    table.innerHTML = `
      <caption>Structured pruning and low-rank comparison on ${escapeHtml(metric.label)}</caption>
      <thead>
        <tr>
          <th scope="col">Category</th>
          <th scope="col">Method</th>
          <th scope="col">Training</th>
          ${comparison.ratios.map((ratio) => `<th scope="col">${ratio}</th>`).join("")}
        </tr>
      </thead>
      <tbody>${body}</tbody>
    `;
    document.querySelector("#comparison-status").textContent =
      `Llama-3.1-8B · ${metric.label}`;
  };

  const buildProfileTabs = () => {
    const container = document.querySelector("#profile-tabs");

    Object.entries(data.serving.profiles).forEach(([key, profile]) => {
      const active = key === state.profile;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.profile = key;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(active));
      button.setAttribute("aria-pressed", String(active));
      button.tabIndex = active ? 0 : -1;
      button.classList.toggle("is-active", active);
      button.innerHTML = `<strong>${escapeHtml(profile.label)}</strong><span>${escapeHtml(profile.tokens)}</span>`;
      button.addEventListener("click", () => {
        state.profile = key;
        setPressed([...container.querySelectorAll("button")], button);
        renderServing();
      });
      container.append(button);
    });
  };

  const renderServing = () => {
    const table = document.querySelector("#serving-table");
    const rows = data.serving.rows;
    const directions = ["min", "min", "max", "max"];
    const svdRows = rows.filter((row) => row.group === "svd");
    const best = directions.map((direction, index) => {
      const values = svdRows.map((row) => row[state.profile][index]);
      return direction === "min" ? Math.min(...values) : Math.max(...values);
    });

    const body = rows
      .map((row, rowIndex) => {
        const classes = [
          row.group === "reference" ? "dense-row" : "",
          row.group === "audit" ? "audit-row" : "",
          rowIndex === 1 || row.group === "audit" && rows[rowIndex - 1].group !== "audit"
            ? "group-start"
            : ""
        ]
          .filter(Boolean)
          .join(" ");
        const cells = row[state.profile]
          .map((value, index) => {
            const isBest = row.group === "svd" && value === best[index];
            return `<td${isBest ? ' class="best-cell"' : ""}>${value.toFixed(2)}</td>`;
          })
          .join("");
        return `<tr${classes ? ` class="${classes}"` : ""}><td>${escapeHtml(row.method)}</td>${cells}</tr>`;
      })
      .join("");

    const profile = data.serving.profiles[state.profile];
    table.innerHTML = `
      <caption>${escapeHtml(profile.label)} serving results, ${escapeHtml(profile.tokens)}</caption>
      <thead>
        <tr>
          <th scope="col">Method</th>
          <th scope="col">TTFT ms ↓</th>
          <th scope="col">E2E ms ↓</th>
          <th scope="col">Input tok/s ↑</th>
          <th scope="col">Output tok/s ↑</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    `;
    document.querySelector("#serving-status").textContent =
      `${profile.label} · ${profile.tokens} tokens · A100 / BF16`;
  };

  const bindSegmentedControls = () => {
    const keepButtons = [...document.querySelectorAll("[data-keep]")];
    keepButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.keep = button.dataset.keep;
        setPressed(keepButtons, button);
        renderAccuracy();
      });
    });

    const viewButtons = [...document.querySelectorAll("[data-view]")];
    viewButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.view = button.dataset.view;
        setPressed(viewButtons, button);
        renderAccuracy();
      });
    });

    const comparisonButtons = [...document.querySelectorAll("[data-comparison]")];
    comparisonButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.comparison = button.dataset.comparison;
        setPressed(comparisonButtons, button);
        renderComparison();
      });
    });
  };

  const copyCitation = async () => {
    const button = document.querySelector("#copy-citation");
    const citation = document.querySelector("#citation-code").textContent;

    try {
      await navigator.clipboard.writeText(citation);
    } catch {
      const textArea = document.createElement("textarea");
      textArea.value = citation;
      textArea.setAttribute("readonly", "");
      textArea.style.position = "fixed";
      textArea.style.opacity = "0";
      document.body.append(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
    }

    button.textContent = "Copied";
    button.classList.add("is-copied");
    window.setTimeout(() => {
      button.textContent = "Copy BibTeX";
      button.classList.remove("is-copied");
    }, 1600);
  };

  buildModelTabs();
  buildProfileTabs();
  bindSegmentedControls();
  renderAccuracy();
  renderComparison();
  renderServing();
  document.querySelector("#copy-citation").addEventListener("click", copyCitation);
})();
