const state = {
  session: null,
  itemKey: null,
  selectedTopic: null,
  previewFrames: [],
  currentFrame: 0,
  breakpoints: [],
  outputPaths: [],
  playing: false,
  playbackRate: Number(localStorage.getItem("mcapSplitPlaybackRate")) || 1,
  playTimer: null,
  exporting: false,
  sessionTimer: null,
  previewTimer: null,
  splitTimer: null,
};

const elements = {
  headerStatus: document.querySelector("#header-status"),
  sourcePath: document.querySelector("#source-path"),
  totalItems: document.querySelector("#total-items"),
  currentItem: document.querySelector("#current-item"),
  completedItems: document.querySelector("#completed-items"),
  skippedItems: document.querySelector("#skipped-items"),
  batchProgressBar: document.querySelector("#batch-progress-bar"),
  currentSection: document.querySelector("#current-section"),
  currentTitle: document.querySelector("#current-title"),
  currentPath: document.querySelector("#current-path"),
  skipButton: document.querySelector("#skip-button"),
  scanProgress: document.querySelector("#scan-progress"),
  scanProgressMessage: document.querySelector("#scan-progress-message"),
  scanProgressPercent: document.querySelector("#scan-progress-percent"),
  scanProgressBar: document.querySelector("#scan-progress-bar"),
  itemContent: document.querySelector("#item-content"),
  itemStats: document.querySelector("#item-stats"),
  cameraCount: document.querySelector("#camera-count"),
  cameraGrid: document.querySelector("#camera-grid"),
  cameraMessage: document.querySelector("#camera-message"),
  previewSection: document.querySelector("#preview-section"),
  selectedTopic: document.querySelector("#selected-topic"),
  previewLoading: document.querySelector("#preview-loading"),
  previewProgressMessage: document.querySelector("#preview-progress-message"),
  previewProgressPercent: document.querySelector("#preview-progress-percent"),
  previewProgressBar: document.querySelector("#preview-progress-bar"),
  player: document.querySelector("#player"),
  previewImage: document.querySelector("#preview-image"),
  frameBadge: document.querySelector("#frame-badge"),
  playButton: document.querySelector("#play-button"),
  playbackRate: document.querySelector("#playback-rate"),
  timeline: document.querySelector("#timeline"),
  timelineMarkers: document.querySelector("#timeline-markers"),
  timeValue: document.querySelector("#time-value"),
  addBreakpoint: document.querySelector("#add-breakpoint"),
  breakpointMessage: document.querySelector("#breakpoint-message"),
  segmentsSection: document.querySelector("#segments-section"),
  segmentCount: document.querySelector("#segment-count"),
  batchSegmentWarning: document.querySelector("#batch-segment-warning"),
  breakpointList: document.querySelector("#breakpoint-list"),
  splitForm: document.querySelector("#split-form"),
  segmentList: document.querySelector("#segment-list"),
  exportProgress: document.querySelector("#export-progress"),
  exportProgressMessage: document.querySelector("#export-progress-message"),
  exportProgressPercent: document.querySelector("#export-progress-percent"),
  exportProgressBar: document.querySelector("#export-progress-bar"),
  exportMessage: document.querySelector("#export-message"),
  splitButton: document.querySelector("#split-button"),
  historyList: document.querySelector("#history-list"),
  doneSection: document.querySelector("#done-section"),
  doneSummary: document.querySelector("#done-summary"),
  cameraTemplate: document.querySelector("#camera-template"),
  segmentTemplate: document.querySelector("#segment-template"),
};

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(value || 0);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 2)} ${units[index]}`;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(3).padStart(6, "0")}`;
}

function frameRelativeSeconds(frameIndex) {
  if (!state.previewFrames.length) return 0;
  const current = BigInt(state.previewFrames[frameIndex].timestamp_ns);
  const first = BigInt(state.previewFrames[0].timestamp_ns);
  return Number(current - first) / 1e9;
}

function setHeaderStatus(label, mode = "ready") {
  elements.headerStatus.replaceChildren();
  const dot = document.createElement("span");
  dot.className = `status-dot ${mode === "loading" ? "is-loading" : ""} ${mode === "error" ? "is-error" : ""}`;
  const text = document.createElement("span");
  text.textContent = label;
  elements.headerStatus.append(dot, text);
}

function updateProgress(prefix, status) {
  const progress = Math.max(0, Math.min(100, Number(status.progress) || 0));
  elements[`${prefix}ProgressMessage`].textContent = status.message || "正在处理";
  elements[`${prefix}ProgressPercent`].textContent = `${progress.toFixed(progress % 1 ? 1 : 0)}%`;
  elements[`${prefix}ProgressBar`].style.width = `${progress}%`;
}

function resetItemUi() {
  stopPlayback();
  state.selectedTopic = null;
  state.previewFrames = [];
  state.currentFrame = 0;
  state.breakpoints = [];
  state.outputPaths = [];
  state.exporting = false;
  elements.itemContent.hidden = true;
  elements.previewSection.hidden = true;
  elements.segmentsSection.hidden = true;
  elements.player.hidden = true;
  elements.exportProgress.hidden = true;
  elements.exportMessage.textContent = "";
  elements.cameraMessage.textContent = "";
  clearTimeout(state.previewTimer);
  clearTimeout(state.splitTimer);
}

function renderBatch(session) {
  const completeCount = session.history.filter((item) => item.state === "completed").length;
  const skippedCount = session.history.filter((item) => item.state === "skipped").length;
  elements.sourcePath.textContent = session.source;
  elements.sourcePath.title = session.source;
  elements.totalItems.textContent = formatNumber(session.total_items);
  elements.currentItem.textContent = session.completed ? "完成" : `${session.current_number} / ${session.total_items}`;
  elements.completedItems.textContent = formatNumber(completeCount);
  elements.skippedItems.textContent = formatNumber(skippedCount);
  const processed = completeCount + skippedCount;
  elements.batchProgressBar.style.width = `${session.total_items ? (processed / session.total_items) * 100 : 100}%`;
  renderHistory(session.history);
}

function renderHistory(history) {
  elements.historyList.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "尚无处理记录";
    elements.historyList.append(empty);
    return;
  }
  history.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-row";
    const status = document.createElement("span");
    status.className = `history-state ${item.state === "skipped" ? "is-skipped" : ""}`;
    status.textContent = item.state === "skipped" ? "已跳过" : "已完成";
    const path = document.createElement("span");
    path.className = "history-path";
    path.textContent = item.source;
    path.title = item.source;
    const outputs = document.createElement("span");
    outputs.className = "history-outputs";
    outputs.textContent = item.outputs.length ? `${item.outputs.length} 个片段` : "—";
    outputs.title = item.outputs.join("\n");
    row.append(status, path, outputs);
    elements.historyList.append(row);
  });
}

async function loadSession() {
  clearTimeout(state.sessionTimer);
  try {
    const response = await fetch("/api/session");
    const session = await response.json();
    if (!response.ok) throw new Error(session.error || "无法读取处理状态");
    state.session = session;
    renderBatch(session);
    if (session.completed) {
      resetItemUi();
      elements.currentSection.hidden = true;
      elements.doneSection.hidden = false;
      const completed = session.history.filter((item) => item.state === "completed").length;
      const skipped = session.history.filter((item) => item.state === "skipped").length;
      elements.doneSummary.textContent = `已完成 ${completed} 条，跳过 ${skipped} 条。`;
      setHeaderStatus("全部完成");
      return;
    }

    elements.currentSection.hidden = false;
    elements.doneSection.hidden = true;
    const key = `${session.current_index}:${session.item.state}`;
    if (state.itemKey === null || !state.itemKey.startsWith(`${session.current_index}:`)) {
      state.itemKey = key;
      resetItemUi();
    }
    elements.currentTitle.textContent = `当前数据 · ${session.current_number}/${session.total_items}`;
    updateProgress("scan", session.item);
    if (session.item.state === "failed") {
      setHeaderStatus("解析失败", "error");
      elements.currentPath.textContent = session.item.error || session.item.message;
      return;
    }
    if (session.item.state !== "completed") {
      setHeaderStatus("正在解析", "loading");
      state.sessionTimer = setTimeout(loadSession, 400);
      return;
    }

    const item = session.item.result;
    elements.currentPath.textContent = item.path;
    elements.currentPath.title = item.path;
    elements.scanProgress.hidden = true;
    renderItem(item);
    setHeaderStatus("等待选择相机");
  } catch (error) {
    setHeaderStatus("读取失败", "error");
    elements.currentPath.textContent = error.message;
    state.sessionTimer = setTimeout(loadSession, 1500);
  }
}

function renderItem(item) {
  if (!elements.itemContent.hidden) return;
  elements.itemContent.hidden = false;
  elements.itemStats.replaceChildren();
  [
    `${formatNumber(item.messages)} 条消息`,
    `${item.duration_s.toFixed(3)} 秒`,
    formatBytes(item.size_bytes),
  ].forEach((value) => {
    const badge = document.createElement("span");
    badge.textContent = value;
    elements.itemStats.append(badge);
  });
  elements.cameraCount.textContent = `${item.cameras.length} 个相机`;
  elements.cameraGrid.replaceChildren();
  item.cameras.forEach((camera) => {
    const card = elements.cameraTemplate.content.firstElementChild.cloneNode(true);
    card.dataset.topic = camera.topic;
    const image = card.querySelector("img");
    image.src = `/api/camera/thumbnail?topic=${encodeURIComponent(camera.topic)}&item=${state.session.current_index}`;
    card.querySelector(".camera-topic").textContent = camera.topic;
    card.querySelector(".camera-schema").textContent = camera.schema;
    const [height, width] = camera.shape;
    card.querySelector(".camera-metrics").textContent = `${width} × ${height} · ${camera.fps.toFixed(2)} FPS · ${formatNumber(camera.frames)} 帧`;
    card.addEventListener("click", () => selectCamera(camera.topic));
    elements.cameraGrid.append(card);
  });
  if (!item.cameras.length) {
    elements.cameraMessage.textContent = "当前 MCAP 没有可解码的相机数据。";
  } else if (item.warnings.length) {
    elements.cameraMessage.textContent = `有 ${item.warnings.length} 个相机 topic 无法预览。`;
  }
  const preferredTopic = state.session.preferred_camera_topic;
  const itemIndex = state.session.current_index;
  if (preferredTopic && item.cameras.some((camera) => camera.topic === preferredTopic)) {
    window.setTimeout(() => {
      if (state.selectedTopic === null && state.session.current_index === itemIndex) {
        selectCamera(preferredTopic);
      }
    }, 0);
  } else if (preferredTopic && item.cameras.length) {
    elements.cameraMessage.textContent = `首选相机 ${preferredTopic} 在当前数据中不可用，请重新选择。`;
  }
}

async function selectCamera(topic) {
  if (state.exporting) return;
  stopPlayback();
  state.selectedTopic = topic;
  state.previewFrames = [];
  state.breakpoints = [];
  state.outputPaths = [];
  elements.previewSection.hidden = false;
  elements.selectedTopic.textContent = topic;
  elements.selectedTopic.title = topic;
  elements.previewLoading.hidden = false;
  elements.player.hidden = true;
  elements.segmentsSection.hidden = true;
  elements.breakpointMessage.textContent = "";
  document.querySelectorAll(".camera-card").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.topic === topic);
  });
  updateProgress("preview", {progress: 0, message: "正在准备相机预览"});
  setHeaderStatus("正在准备预览", "loading");
  try {
    const response = await fetch("/api/preview", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_index: state.session.current_index, topic}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "无法准备相机预览");
    pollPreview();
  } catch (error) {
    elements.cameraMessage.textContent = error.message;
    setHeaderStatus("预览失败", "error");
  }
}

async function pollPreview() {
  clearTimeout(state.previewTimer);
  try {
    const response = await fetch("/api/preview/status");
    const status = await response.json();
    updateProgress("preview", status);
    if (status.state === "failed") throw new Error(status.error || status.message || "预览失败");
    if (status.state !== "completed") {
      state.previewTimer = setTimeout(pollPreview, 400);
      return;
    }
    const timelineResponse = await fetch("/api/preview/timeline");
    const timeline = await timelineResponse.json();
    if (!timelineResponse.ok) throw new Error(timeline.error || "无法读取相机时间轴");
    state.previewFrames = timeline.frames;
    state.currentFrame = 0;
    elements.previewLoading.hidden = true;
    elements.player.hidden = false;
    elements.segmentsSection.hidden = false;
    elements.timeline.max = String(state.previewFrames.length - 1);
    elements.timeline.value = "0";
    renderCurrentFrame();
    renderSegments();
    setHeaderStatus("等待添加断点");
  } catch (error) {
    elements.cameraMessage.textContent = error.message;
    setHeaderStatus("预览失败", "error");
  }
}

function renderCurrentFrame() {
  if (!state.previewFrames.length) return;
  elements.previewImage.src = `/api/preview/frame?index=${state.currentFrame}`;
  elements.frameBadge.textContent = `${state.currentFrame + 1} / ${state.previewFrames.length}`;
  elements.timeValue.textContent = formatTime(frameRelativeSeconds(state.currentFrame));
  elements.timeline.value = String(state.currentFrame);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    return;
  }
  state.playing = true;
  elements.playButton.textContent = "Ⅱ";
  elements.playButton.title = "暂停";
  startPlaybackTimer();
}

function startPlaybackTimer() {
  clearInterval(state.playTimer);
  const selectedCamera = state.session.item.result.cameras.find(
    (camera) => camera.topic === state.selectedTopic,
  );
  const fps = Number(selectedCamera ? selectedCamera.fps : 0) || 30;
  const delay = Math.max(1, Math.round(1000 / fps / state.playbackRate));
  state.playTimer = setInterval(() => {
    if (state.currentFrame >= state.previewFrames.length - 1) {
      stopPlayback();
      return;
    }
    state.currentFrame += 1;
    renderCurrentFrame();
  }, delay);
}

function stopPlayback() {
  clearInterval(state.playTimer);
  state.playTimer = null;
  state.playing = false;
  elements.playButton.textContent = "▶";
  elements.playButton.title = "播放";
}

function addBreakpoint() {
  elements.breakpointMessage.textContent = "";
  if (state.currentFrame === 0) {
    elements.breakpointMessage.textContent = "首帧不能作为断点。";
    return;
  }
  const timestamp = state.previewFrames[state.currentFrame].timestamp_ns;
  if (state.breakpoints.some((item) => item.timestamp_ns === timestamp)) {
    elements.breakpointMessage.textContent = "当前位置已经存在断点。";
    return;
  }
  state.breakpoints.push({frame_index: state.currentFrame, timestamp_ns: timestamp});
  state.breakpoints.sort((left, right) => left.frame_index - right.frame_index);
  renderSegments();
}

function defaultOutputPath(index) {
  return state.session.item.result.suggested_output_pattern.replace("{segment}", String(index + 1));
}

function renderSegments() {
  const segmentTotal = state.breakpoints.length + 1;
  state.outputPaths = Array.from(
    {length: segmentTotal},
    (_, index) => state.outputPaths[index] || defaultOutputPath(index),
  );
  elements.segmentCount.textContent = `${segmentTotal} 个片段`;
  const expectedSegments = state.session.expected_segments;
  const differsFromBatch = expectedSegments !== null && segmentTotal !== expectedSegments;
  elements.batchSegmentWarning.hidden = !differsFromBatch;
  elements.batchSegmentWarning.textContent = differsFromBatch
    ? `本批次默认导出 ${expectedSegments} 个片段，当前设置为 ${segmentTotal} 个。导出前需要确认。`
    : "";
  elements.breakpointList.replaceChildren();
  state.breakpoints.forEach((breakpoint, index) => {
    const chip = document.createElement("span");
    chip.className = "breakpoint-chip";
    const label = document.createElement("span");
    label.textContent = `断点 ${index + 1} · ${formatTime(frameRelativeSeconds(breakpoint.frame_index))}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "移除断点";
    remove.setAttribute("aria-label", `移除断点 ${index + 1}`);
    remove.addEventListener("click", () => {
      state.breakpoints.splice(index, 1);
      renderSegments();
    });
    chip.append(label, remove);
    elements.breakpointList.append(chip);
  });

  elements.timelineMarkers.replaceChildren();
  state.breakpoints.forEach((breakpoint) => {
    const marker = document.createElement("span");
    marker.className = "timeline-marker";
    marker.style.left = `${(breakpoint.frame_index / (state.previewFrames.length - 1)) * 100}%`;
    elements.timelineMarkers.append(marker);
  });

  elements.segmentList.replaceChildren();
  for (let index = 0; index < segmentTotal; index += 1) {
    const row = elements.segmentTemplate.content.firstElementChild.cloneNode(true);
    row.querySelector(".segment-index").textContent = `片段 ${index + 1}`;
    const start = index === 0 ? 0 : frameRelativeSeconds(state.breakpoints[index - 1].frame_index);
    const end = index === state.breakpoints.length
      ? frameRelativeSeconds(state.previewFrames.length - 1)
      : frameRelativeSeconds(state.breakpoints[index].frame_index);
    row.querySelector(".segment-range").textContent = `${formatTime(start)} → ${formatTime(end)}`;
    const input = row.querySelector(".segment-path");
    input.value = state.outputPaths[index];
    input.addEventListener("input", () => {
      state.outputPaths[index] = input.value.trim();
      validateSplit();
    });
    elements.segmentList.append(row);
  }
  validateSplit();
}

function validateSplit() {
  const paths = state.outputPaths.map((path) => path.trim());
  const valid =
    state.breakpoints.length > 0 &&
    paths.length === state.breakpoints.length + 1 &&
    paths.every((path) => path.toLowerCase().endsWith(".mcap")) &&
    new Set(paths).size === paths.length;
  elements.splitButton.disabled = !valid || state.exporting;
}

async function startSplit(event) {
  event.preventDefault();
  validateSplit();
  if (elements.splitButton.disabled) return;
  const expectedSegments = state.session.expected_segments;
  const segmentTotal = state.breakpoints.length + 1;
  let confirmedSegmentCount = false;
  if (expectedSegments !== null && segmentTotal !== expectedSegments) {
    confirmedSegmentCount = window.confirm(
      `本批次默认分割为 ${expectedSegments} 个片段，当前为 ${segmentTotal} 个。确认按当前数量导出吗？`,
    );
    if (!confirmedSegmentCount) return;
  }
  stopPlayback();
  state.exporting = true;
  validateSplit();
  elements.exportProgress.hidden = false;
  elements.exportMessage.textContent = "";
  elements.exportMessage.className = "export-message";
  updateProgress("export", {progress: 0, message: "正在准备分割"});
  setHeaderStatus("正在导出", "loading");
  try {
    const payload = {
      item_index: state.session.current_index,
      topic: state.selectedTopic,
      breakpoints_ns: state.breakpoints.map((item) => item.timestamp_ns),
      output_paths: state.outputPaths,
      confirm_segment_count: confirmedSegmentCount,
      overwrite_existing: false,
    };
    let response = await fetch("/api/split", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    let result = await response.json();
    if (response.status === 409 && result.code === "outputs_exist") {
      const paths = Array.isArray(result.paths) ? result.paths : [];
      const visiblePaths = paths.slice(0, 5).join("\n");
      const remaining = paths.length > 5 ? `\n……另有 ${paths.length - 5} 个文件` : "";
      const confirmedOverwrite = window.confirm(
        `发现 ${paths.length} 个已存在的输出文件，是否覆盖？\n\n${visiblePaths}${remaining}`,
      );
      if (!confirmedOverwrite) {
        cancelSplit("已取消覆盖，现有文件未被修改。");
        return;
      }
      payload.overwrite_existing = true;
      response = await fetch("/api/split", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      result = await response.json();
    }
    if (!response.ok) throw new Error(result.error || "无法启动分割任务");
    pollSplit();
  } catch (error) {
    finishSplit(false, error.message);
  }
}

function cancelSplit(message) {
  state.exporting = false;
  validateSplit();
  elements.exportProgress.hidden = true;
  elements.exportMessage.textContent = message;
  elements.exportMessage.className = "export-message is-neutral";
  setHeaderStatus("等待导出");
}

async function pollSplit() {
  clearTimeout(state.splitTimer);
  try {
    const response = await fetch("/api/split/status");
    const status = await response.json();
    updateProgress("export", status);
    if (status.state === "failed") {
      finishSplit(false, status.error || status.message || "分割失败");
      return;
    }
    if (status.state !== "completed") {
      state.splitTimer = setTimeout(pollSplit, 500);
      return;
    }
    finishSplit(true, `已导出 ${status.result.segments} 个 MCAP 片段，正在进入下一条数据。`);
    state.itemKey = null;
    setTimeout(loadSession, 450);
  } catch (error) {
    state.splitTimer = setTimeout(pollSplit, 1200);
  }
}

function finishSplit(success, message) {
  state.exporting = false;
  validateSplit();
  elements.exportMessage.textContent = message;
  elements.exportMessage.className = `export-message ${success ? "is-success" : ""}`;
  setHeaderStatus(success ? "导出完成" : "导出失败", success ? "ready" : "error");
}

async function skipCurrent() {
  if (!state.session || state.exporting) return;
  stopPlayback();
  elements.skipButton.disabled = true;
  try {
    const response = await fetch("/api/item/skip", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_index: state.session.current_index}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "无法跳过当前数据");
    state.itemKey = null;
    await loadSession();
  } catch (error) {
    elements.cameraMessage.textContent = error.message;
    setHeaderStatus("跳过失败", "error");
  } finally {
    elements.skipButton.disabled = false;
  }
}

elements.playButton.addEventListener("click", togglePlayback);
elements.playbackRate.value = String(state.playbackRate);
elements.playbackRate.addEventListener("change", () => {
  state.playbackRate = Number(elements.playbackRate.value) || 1;
  localStorage.setItem("mcapSplitPlaybackRate", String(state.playbackRate));
  if (state.playing) startPlaybackTimer();
});
elements.timeline.addEventListener("input", () => {
  stopPlayback();
  state.currentFrame = Number(elements.timeline.value);
  renderCurrentFrame();
});
elements.addBreakpoint.addEventListener("click", addBreakpoint);
elements.splitForm.addEventListener("submit", startSplit);
elements.skipButton.addEventListener("click", skipCurrent);

loadSession();
