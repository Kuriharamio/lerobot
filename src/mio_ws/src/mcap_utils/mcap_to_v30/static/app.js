const state = {
  scan: null,
  scanPollTimer: null,
  selectedTopics: [],
  mappings: [],
  exporting: false,
  pollTimer: null,
};

const elements = {
  headerStatus: document.querySelector("#header-status"),
  sourcePath: document.querySelector("#source-path"),
  statGrid: document.querySelector("#stat-grid"),
  metadataDetails: document.querySelector("#metadata-details"),
  scanProgress: document.querySelector("#scan-progress"),
  scanProgressMessage: document.querySelector("#scan-progress-message"),
  scanProgressPercent: document.querySelector("#scan-progress-percent"),
  scanProgressBar: document.querySelector("#scan-progress-bar"),
  scanDiagnostics: document.querySelector("#scan-diagnostics"),
  topicSummary: document.querySelector("#topic-summary"),
  topicSearch: document.querySelector("#topic-search"),
  topicList: document.querySelector("#topic-list"),
  selectionCount: document.querySelector("#selection-count"),
  selectionPreview: document.querySelector("#selection-preview"),
  mappingName: document.querySelector("#mapping-name"),
  mergeButton: document.querySelector("#merge-button"),
  mappingError: document.querySelector("#mapping-error"),
  mappingList: document.querySelector("#mapping-list"),
  mappingSummary: document.querySelector("#mapping-summary"),
  exportForm: document.querySelector("#export-form"),
  exportButton: document.querySelector("#export-button"),
  fps: document.querySelector("#fps"),
  repoId: document.querySelector("#repo-id"),
  root: document.querySelector("#root"),
  task: document.querySelector("#task"),
  robotType: document.querySelector("#robot-type"),
  useVideos: document.querySelector("#use-videos"),
  pushToHub: document.querySelector("#push-to-hub"),
  private: document.querySelector("#private"),
  resolvedPath: document.querySelector("#resolved-path"),
  exportProgress: document.querySelector("#export-progress"),
  progressMessage: document.querySelector("#progress-message"),
  progressPercent: document.querySelector("#progress-percent"),
  progressBar: document.querySelector("#progress-bar"),
  exportMessage: document.querySelector("#export-message"),
  topicTemplate: document.querySelector("#topic-template"),
  mappingTemplate: document.querySelector("#mapping-template"),
};

function escapeText(value) {
  return String(value ?? "");
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 2)} ${units[index]}`;
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = Math.floor(seconds % 60);
  return hours ? `${hours}h ${minutes}m ${rest}s` : `${minutes}m ${rest}s`;
}

function formatShape(shape) {
  return Array.isArray(shape) && shape.length ? `[${shape.join(" × ")}]` : "shape —";
}

function lerobotShape(mapping) {
  const topics = mapping.topics.map((name) => state.scan.topics.find((topic) => topic.name === name));
  if (topics.some((topic) => !topic || !Array.isArray(topic.shape))) return null;
  if (topics.length === 1 && topics[0].kind === "image" && topics[0].shape.length === 3) {
    const [height, width, channels] = topics[0].shape;
    return [channels, height, width];
  }
  if (topics.every((topic) => topic.shape.length === 1 && topic.kind !== "image")) {
    return [topics.reduce((total, topic) => total + topic.shape[0], 0)];
  }
  return null;
}

function mappingTopics(mapping) {
  return mapping.topics.map((name) => state.scan.topics.find((topic) => topic.name === name));
}

function isImageMapping(mapping) {
  const topics = mappingTopics(mapping);
  return topics.length === 1 && topics[0]?.kind === "image";
}

function genericTopicNames(topic) {
  if (!Array.isArray(topic.shape) || topic.shape.length !== 1) return [];
  const prefix = topic.name.replace(/^\/+|\/+$/g, "").replaceAll("/", ".") || "value";
  return Array.from({length: topic.shape[0]}, (_, index) => `${prefix}.dim_${index + 1}`);
}

function automaticMappingNames(mapping) {
  return mappingTopics(mapping).flatMap((topic) => {
    if (!topic) return [];
    if (Array.isArray(topic.names) && topic.names.length === topic.shape?.[0]) {
      return [...topic.names];
    }
    return genericTopicNames(topic);
  });
}

function parseMappingNames(value) {
  if (!value.trim()) return [];
  return value.replaceAll("\r", "").split("\n").map((name) => name.trim());
}

function mappingNamesValidation(mapping) {
  if (isImageMapping(mapping)) return {valid: true, expected: 0, count: 0, message: ""};
  const shape = lerobotShape(mapping);
  const expected = shape?.length === 1 ? shape[0] : 0;
  const names = Array.isArray(mapping.names) ? mapping.names : [];
  if (!expected) {
    return {valid: false, expected, count: names.length, message: "无法确定合并后的向量维度"};
  }
  if (names.length !== expected) {
    return {
      valid: false,
      expected,
      count: names.length,
      message: `需要 ${expected} 个名称，当前为 ${names.length} 个`,
    };
  }
  if (names.some((name) => !name)) {
    return {valid: false, expected, count: names.length, message: "维度名称不能为空"};
  }
  if (new Set(names).size !== names.length) {
    return {valid: false, expected, count: names.length, message: "维度名称不能重复"};
  }
  return {valid: true, expected, count: names.length, message: ""};
}

function setHeaderStatus(label, mode = "ready") {
  elements.headerStatus.replaceChildren();
  const dot = document.createElement("span");
  dot.className = `status-dot ${mode === "loading" ? "is-loading" : ""} ${mode === "error" ? "is-error" : ""}`;
  const text = document.createElement("span");
  text.textContent = label;
  elements.headerStatus.append(dot, text);
}

function renderStats(scan) {
  const stats = [
    ["MCAP 文件", formatNumber(scan.file_count)],
    ["Topics", formatNumber(scan.topics.length)],
    ["消息总数", formatNumber(scan.total_messages)],
    ["累计时长", formatDuration(scan.total_duration_s)],
    ["数据大小", formatBytes(scan.total_size_bytes)],
  ];
  elements.statGrid.replaceChildren();
  stats.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "stat-item";
    const caption = document.createElement("span");
    caption.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    item.append(caption, strong);
    elements.statGrid.append(item);
  });
}

function metadataValue(field) {
  if (field.variant_count <= 1) return field.value || "—";
  return field.variants
    .map((variant) => `${variant.value}  [${variant.files.join(", ")}]`)
    .join("\n");
}

function renderMetadata(groups) {
  elements.metadataDetails.replaceChildren();
  groups.forEach((group) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${group.name} · ${group.fields.length}`;
    const table = document.createElement("dl");
    table.className = "metadata-table";
    group.fields.forEach((field) => {
      const row = document.createElement("div");
      row.className = "metadata-field";
      const key = document.createElement("dt");
      key.textContent = field.key;
      const value = document.createElement("dd");
      value.textContent = metadataValue(field);
      row.append(key, value);
      table.append(row);
    });
    details.append(summary, table);
    elements.metadataDetails.append(details);
  });
}

function updateScanProgress(status) {
  const progress = Math.max(0, Math.min(100, Number(status.progress) || 0));
  let message = status.message || "正在解析 MCAP 数据";
  if (status.file && status.files) message += ` · ${status.file}/${status.files}`;
  elements.scanProgress.hidden = false;
  elements.scanProgressMessage.textContent = message;
  elements.scanProgressPercent.textContent = `${progress.toFixed(progress % 1 ? 1 : 0)}%`;
  elements.scanProgressBar.style.width = `${progress}%`;
}

function renderScanDiagnostics(inventory, fileCount) {
  const missing = inventory?.missing_episode_paths || [];
  elements.scanDiagnostics.replaceChildren();
  elements.scanDiagnostics.hidden = missing.length === 0;
  if (!missing.length) return;

  const title = document.createElement("strong");
  title.textContent = `Episode 编号存在断档：预计 ${inventory.expected_episodes} 个，找到 ${fileCount} 个 MCAP`;
  elements.scanDiagnostics.append(title);
  missing.forEach((path) => {
    const item = document.createElement("code");
    item.textContent = `缺少 ${path}/episode.mcap`;
    elements.scanDiagnostics.append(item);
  });
}

function renderTopics() {
  if (!state.scan) return;
  const query = elements.topicSearch.value.trim().toLowerCase();
  const topics = state.scan.topics.filter((topic) => topic.name.toLowerCase().includes(query));
  elements.topicList.replaceChildren();
  topics.forEach((topic) => {
    const row = elements.topicTemplate.content.firstElementChild.cloneNode(true);
    const selectedIndex = state.selectedTopics.indexOf(topic.name);
    row.classList.toggle("is-selected", selectedIndex >= 0);
    row.querySelector(".order-box").textContent = selectedIndex >= 0 ? selectedIndex + 1 : "";
    row.querySelector(".topic-name").textContent = topic.name;
    row.querySelector(".topic-schema").textContent = `${topic.schema} · ${topic.files}/${state.scan.file_count} 文件`;
    row.querySelector(".shape-value").textContent = formatShape(topic.shape);
    row.querySelector(".fps-value").textContent = `${topic.fps.toFixed(2)} FPS`;
    row.querySelector(".frame-value").textContent = `${formatNumber(topic.frames)} 帧`;
    row.addEventListener("click", () => toggleTopic(topic.name));
    elements.topicList.append(row);
  });
}

function toggleTopic(topicName) {
  const index = state.selectedTopics.indexOf(topicName);
  if (index >= 0) {
    state.selectedTopics.splice(index, 1);
  } else {
    state.selectedTopics.push(topicName);
  }
  renderTopics();
  renderSelection();
}

function renderSelection() {
  const count = state.selectedTopics.length;
  elements.selectionCount.textContent = `已选择 ${count} 项`;
  elements.selectionPreview.textContent = count
    ? state.selectedTopics.map((topic, index) => `${index + 1}. ${topic}`).join("   +   ")
    : "未选择 topic";
  elements.mergeButton.disabled = !count || !elements.mappingName.value.trim();
}

function addMapping() {
  const target = elements.mappingName.value.trim();
  elements.mappingError.textContent = "";
  if (!state.selectedTopics.length || !target) return;
  if (state.mappings.some((mapping) => mapping.target === target)) {
    elements.mappingError.textContent = `映射名称 ${target} 已存在`;
    return;
  }
  const selectedRecords = state.selectedTopics.map((name) => state.scan.topics.find((topic) => topic.name === name));
  if (selectedRecords.some((topic) => topic.kind === "image") && selectedRecords.length !== 1) {
    elements.mappingError.textContent = "图像 topic 需要单独映射，不能与其他 topic 合并";
    return;
  }
  const mapping = { target, topics: [...state.selectedTopics] };
  if (!isImageMapping(mapping)) mapping.names = automaticMappingNames(mapping);
  state.mappings.push(mapping);
  state.selectedTopics = [];
  elements.mappingName.value = "";
  renderTopics();
  renderSelection();
  renderMappings();
  validateExport();
}

function renderMappings() {
  elements.mappingList.replaceChildren();
  elements.mappingList.classList.toggle("empty", !state.mappings.length);
  elements.mappingSummary.textContent = state.mappings.length
    ? `${state.mappings.length} 个 LeRobot feature`
    : "尚未创建映射";
  if (!state.mappings.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    const mark = document.createElement("div");
    mark.className = "empty-mark";
    mark.textContent = "→";
    const text = document.createElement("p");
    text.textContent = "LeRobot features";
    empty.append(mark, text);
    elements.mappingList.append(empty);
    return;
  }
  state.mappings.forEach((mapping, mappingIndex) => {
    const row = elements.mappingTemplate.content.firstElementChild.cloneNode(true);
    row.querySelector(".mapping-target").textContent = mapping.target;
    const targetShape = lerobotShape(mapping);
    row.querySelector(".mapping-shape").textContent = `LeRobot shape ${targetShape ? formatShape(targetShape) : "—"}`;
    const sources = row.querySelector(".mapping-sources");
    mapping.topics.forEach((topic, topicIndex) => {
      const source = document.createElement("span");
      source.textContent = `${topicIndex + 1}. ${topic}`;
      sources.append(source);
    });
    const namesControl = row.querySelector(".mapping-names");
    if (!isImageMapping(mapping)) {
      namesControl.hidden = false;
      const editor = row.querySelector(".mapping-names-editor");
      const count = row.querySelector(".mapping-names-count");
      const error = row.querySelector(".mapping-names-error");
      const updateNamesStatus = () => {
        const validation = mappingNamesValidation(mapping);
        count.textContent = `${validation.count} / ${validation.expected}`;
        count.classList.toggle("is-error", !validation.valid);
        error.textContent = validation.message;
        validateExport();
      };
      editor.value = mapping.names.join("\n");
      editor.addEventListener("input", () => {
        mapping.names = parseMappingNames(editor.value);
        updateNamesStatus();
      });
      row.querySelector(".auto-names-button").addEventListener("click", () => {
        mapping.names = automaticMappingNames(mapping);
        editor.value = mapping.names.join("\n");
        updateNamesStatus();
      });
      updateNamesStatus();
    }
    row.querySelector(".remove-button").addEventListener("click", () => {
      state.mappings.splice(mappingIndex, 1);
      renderMappings();
      validateExport();
    });
    elements.mappingList.append(row);
  });
}

function fillSuggestions(suggestions) {
  elements.fps.value = suggestions.fps;
  elements.repoId.value = suggestions.repo_id;
  elements.root.value = suggestions.root;
  elements.task.value = suggestions.task;
  updateResolvedPath();
}

function updateResolvedPath() {
  elements.resolvedPath.textContent = elements.root.value.trim() || "—";
}

function validateExport() {
  const complete =
    state.mappings.length > 0 &&
    state.mappings.every((mapping) => mappingNamesValidation(mapping).valid) &&
    Number(elements.fps.value) > 0 &&
    elements.repoId.value.trim() &&
    elements.root.value.trim() &&
    elements.task.value.trim();
  elements.exportButton.disabled = !complete || state.exporting;
}

async function loadScan() {
  try {
    const response = await fetch("/api/scan");
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || "MCAP 解析失败");
    elements.sourcePath.textContent = status.source || "—";
    elements.sourcePath.title = status.source || "";
    updateScanProgress(status);
    if (status.state === "failed") throw new Error(status.error || status.message || "MCAP 解析失败");
    if (status.state !== "completed") {
      setHeaderStatus("正在解析", "loading");
      clearTimeout(state.scanPollTimer);
      state.scanPollTimer = setTimeout(loadScan, 350);
      return;
    }

    const payload = status.result;
    if (!payload) throw new Error("解析完成但未返回 MCAP 结果");
    state.scan = payload;
    elements.sourcePath.textContent = payload.source;
    elements.sourcePath.title = payload.source;
    elements.topicSummary.textContent = `${payload.topics.length} topics · 递归读取 ${payload.file_count} 个文件`;
    renderStats(payload);
    renderScanDiagnostics(payload.inventory, payload.file_count);
    renderMetadata(payload.metadata);
    renderTopics();
    fillSuggestions(payload.suggestions);
    validateExport();
    updateScanProgress({progress: 100, message: "MCAP 解析完成"});
    setHeaderStatus("解析完成");
  } catch (error) {
    setHeaderStatus("解析失败", "error");
    elements.topicList.textContent = error.message;
    elements.exportMessage.textContent = error.message;
  }
}

async function startExport(event) {
  event.preventDefault();
  validateExport();
  if (elements.exportButton.disabled) return;
  state.exporting = true;
  validateExport();
  elements.exportMessage.className = "export-message";
  elements.exportMessage.textContent = "";
  elements.exportProgress.hidden = false;
  updateProgress({ progress: 0, message: "正在准备导出" });
  setHeaderStatus("正在导出", "loading");

  const payload = {
    mappings: state.mappings,
    parameters: {
      fps: Number(elements.fps.value),
      repo_id: elements.repoId.value.trim(),
      root: elements.root.value.trim(),
      task: elements.task.value.trim(),
      robot_type: elements.robotType.value.trim(),
      use_videos: elements.useVideos.checked,
      push_to_hub: elements.pushToHub.checked,
      private: elements.private.checked,
    },
  };

  try {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "无法启动导出任务");
    pollExport();
  } catch (error) {
    finishExport(false, error.message);
  }
}

function updateProgress(job) {
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  elements.progressMessage.textContent = job.message || "正在导出";
  elements.progressPercent.textContent = `${progress.toFixed(progress % 1 ? 1 : 0)}%`;
  elements.progressBar.style.width = `${progress}%`;
}

async function pollExport() {
  clearTimeout(state.pollTimer);
  try {
    const response = await fetch("/api/export/status");
    const job = await response.json();
    updateProgress(job);
    if (job.state === "completed") {
      const result = job.result;
      finishExport(true, `已导出 ${formatNumber(result.episodes)} 个 episodes、${formatNumber(result.frames)} 帧到 ${result.root}`);
      return;
    }
    if (job.state === "failed") {
      finishExport(false, job.error || job.message || "导出失败");
      return;
    }
    state.pollTimer = setTimeout(pollExport, 700);
  } catch (error) {
    state.pollTimer = setTimeout(pollExport, 1500);
  }
}

function finishExport(success, message) {
  state.exporting = false;
  validateExport();
  elements.exportMessage.textContent = message;
  elements.exportMessage.className = `export-message ${success ? "is-success" : ""}`;
  setHeaderStatus(success ? "导出完成" : "导出失败", success ? "ready" : "error");
}

elements.topicSearch.addEventListener("input", renderTopics);
elements.mappingName.addEventListener("input", renderSelection);
elements.mappingName.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addMapping();
  }
});
elements.mergeButton.addEventListener("click", addMapping);
elements.exportForm.addEventListener("submit", startExport);
elements.exportForm.addEventListener("input", validateExport);
elements.root.addEventListener("input", updateResolvedPath);
elements.pushToHub.addEventListener("change", () => {
  elements.private.disabled = !elements.pushToHub.checked;
  if (!elements.pushToHub.checked) elements.private.checked = false;
});

loadScan();
