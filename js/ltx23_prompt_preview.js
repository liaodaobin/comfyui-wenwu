import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const EDITED_PROMPT_PREFIX = "__WENWU_LTX23_EDITED__\n";
const RESOLUTIONS = ["832*480 Landscape", "480*832 Portrait", "1280*720 Landscape", "720*1280 Portrait"];

function displayedPrompt(value) {
  const text = String(value || "");
  return text.startsWith(EDITED_PROMPT_PREFIX) ? text.slice(EDITED_PROMPT_PREFIX.length) : text;
}

function localizeInteractiveWidget(widget, label) {
  if (!widget) return;
  widget.label = label;
  widget.disabled = false;
  widget.options = { ...(widget.options || {}), disabled: false };
}

function repairCoreWidgets(node) {
  const duration = node.widgets?.find((widget) => widget.name === "duration_seconds");
  const frameRate = node.widgets?.find((widget) => widget.name === "frame_rate");
  const resolution = node.widgets?.find((widget) => widget.name === "resolution");
  const seed = node.widgets?.find((widget) => widget.name === "seed");
  const legacyResolution = String(frameRate?.value || "");
  const legacySeed = Number(resolution?.value);
  if (RESOLUTIONS.includes(legacyResolution) && Number.isFinite(legacySeed)) {
    setWidgetValue(node, frameRate, 30);
    setWidgetValue(node, resolution, legacyResolution);
    setWidgetValue(node, seed, legacySeed);
  }
  const durationValue = Number(duration?.value);
  const frameRateValue = Number(frameRate?.value);
  const seedValue = Number(seed?.value);
  if (!Number.isFinite(durationValue) || durationValue < 1 || durationValue > 30) setWidgetValue(node, duration, 8);
  if (!Number.isFinite(frameRateValue) || frameRateValue < 1 || frameRateValue > 120) setWidgetValue(node, frameRate, 30);
  if (!RESOLUTIONS.includes(String(resolution?.value || ""))) setWidgetValue(node, resolution, "720*1280 Portrait");
  if (!Number.isFinite(seedValue) || seedValue < 0) setWidgetValue(node, seed, 0);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  for (const element of [widget.inputEl, widget.element].filter(Boolean)) {
    element.style.display = "none";
    element.hidden = true;
  }
}

function setWidgetValue(node, widget, value) {
  if (!widget) return;
  widget.value = value;
  const index = node.widgets?.indexOf(widget);
  if (index >= 0 && node.widgets_values) node.widgets_values[index] = value;
  widget.callback?.(value);
}

function compactMultilineWidget(widget, height = 88) {
  if (!widget) return;
  const layoutHeight = height + 22;
  widget.computeSize = (width) => [width, layoutHeight];
  for (const element of [widget.inputEl, widget.element].filter(Boolean)) {
    element.style.height = `${height}px`;
    element.style.minHeight = `${height}px`;
    element.style.maxHeight = `${height}px`;
    element.style.resize = "none";
  }
}

async function buildPromptOnlyQueue(nodeId) {
  const prompt = structuredClone(await app.graphToPrompt());
  const output = prompt.output || {};
  const keep = new Set();
  const visit = (id) => {
    const key = String(id);
    if (keep.has(key) || !output[key]) return;
    keep.add(key);
    for (const value of Object.values(output[key].inputs || {})) {
      if (Array.isArray(value) && value.length >= 2 && output[String(value[0])]) visit(value[0]);
    }
  };
  visit(nodeId);
  if (!keep.has(String(nodeId))) throw new Error("LTX-2.3 prompt node is missing from the API prompt");

  prompt.output = Object.fromEntries(Object.entries(output).filter(([key]) => keep.has(String(key))));
  if (prompt.workflow?.nodes) {
    prompt.workflow.nodes = prompt.workflow.nodes.filter((node) => keep.has(String(node.id)));
  }
  if (prompt.workflow?.links) {
    prompt.workflow.links = prompt.workflow.links.filter(
      (link) => keep.has(String(link[1])) && keep.has(String(link[3]))
    );
  }
  return prompt;
}

app.registerExtension({
  name: "WenWu.LTX23PromptPanelV4",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "WenWuLTX23PromptEnhancer") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const themeWidget = this.widgets?.find((widget) => widget.name === "video_theme");
      const promptWidget = this.widgets?.find((widget) => widget.name === "optimized_prompt_display");
      const refreshWidget = this.widgets?.find((widget) => widget.name === "refresh_prompt");
      const durationWidget = this.widgets?.find((widget) => widget.name === "duration_seconds");
      const frameRateWidget = this.widgets?.find((widget) => widget.name === "frame_rate");
      const resolutionWidget = this.widgets?.find((widget) => widget.name === "resolution");
      localizeInteractiveWidget(durationWidget, "视频秒数");
      localizeInteractiveWidget(frameRateWidget, "视频帧率");
      localizeInteractiveWidget(resolutionWidget, "视频分辨率");
      compactMultilineWidget(themeWidget, 88);
      hideWidget(promptWidget);
      hideWidget(refreshWidget);
      repairCoreWidgets(this);

      const panel = document.createElement("div");
      panel.style.cssText = "width:100%;height:370px;display:flex;flex-direction:column;gap:8px;box-sizing:border-box;padding:4px 8px;";

      const topControls = document.createElement("div");
      topControls.style.cssText = "display:flex;gap:8px;align-items:center;";
      const editButton = document.createElement("button");
      const status = document.createElement("span");
      editButton.textContent = "\u7f16\u8f91";
      editButton.style.cssText = "height:30px;padding:0 18px;color:#eee;background:#3b3b3b;border:1px solid #666;border-radius:5px;cursor:pointer;";
      status.textContent = "\u5df2\u9501\u5b9a\uff1a\u8fd0\u884c\u65f6\u91cd\u65b0\u8c03\u7528 Llama";
      status.style.cssText = "font-size:12px;color:#aaa;";
      topControls.append(editButton, status);

      const textarea = document.createElement("textarea");
      textarea.readOnly = true;
      textarea.spellcheck = false;
      textarea.placeholder = "\u751f\u6210\u7684\u82f1\u6587 LTX-2.3 \u589e\u5f3a\u63d0\u793a\u8bcd\u4f1a\u663e\u793a\u5728\u8fd9\u91cc";
      textarea.style.cssText = "width:100%;height:280px;resize:none;box-sizing:border-box;padding:10px;color:#ddd;background:#202020;border:1px solid #555;border-radius:6px;font:13px/1.45 sans-serif;outline:none;";
      textarea.value = displayedPrompt(promptWidget?.value);

      const refreshButton = document.createElement("button");
      refreshButton.textContent = "\u5237\u65b0\u63d0\u793a\u8bcd";
      refreshButton.disabled = true;
      refreshButton.style.cssText = "height:30px;padding:0 18px;color:#eee;background:#3b3b3b;border:1px solid #666;border-radius:5px;cursor:not-allowed;opacity:.55;align-self:flex-start;";

      let editing = false;
      let refreshing = false;
      const styleRefreshButton = () => {
        refreshButton.style.cursor = refreshButton.disabled ? "not-allowed" : "pointer";
        refreshButton.style.opacity = refreshButton.disabled ? ".55" : "1";
      };
      const setEditing = (enabled) => {
        editing = enabled && !refreshing;
        setWidgetValue(this, promptWidget, editing ? EDITED_PROMPT_PREFIX + textarea.value : textarea.value);
        textarea.readOnly = !editing;
        editButton.textContent = editing ? "\u9501\u5b9a" : "\u7f16\u8f91";
        status.textContent = editing
          ? "\u4eba\u5de5\u63d0\u793a\u8bcd\u6a21\u5f0f\uff1a\u8df3\u8fc7 Llama"
          : "\u5df2\u9501\u5b9a\uff1a\u8fd0\u884c\u65f6\u91cd\u65b0\u8c03\u7528 Llama";
        textarea.style.borderColor = editing ? "#6ca0dc" : "#555";
        refreshButton.disabled = !editing || refreshing;
        styleRefreshButton();
        if (editing) textarea.focus();
      };
      const setRefreshing = (enabled, message = "") => {
        refreshing = enabled;
        textarea.readOnly = true;
        editButton.disabled = enabled;
        refreshButton.disabled = true;
        editButton.style.cursor = enabled ? "not-allowed" : "pointer";
        editButton.style.opacity = enabled ? ".55" : "1";
        styleRefreshButton();
        status.textContent = message || (enabled ? "\u6b63\u5728\u5237\u65b0\u2026" : "\u5df2\u9501\u5b9a");
      };

      textarea.addEventListener("input", () => {
        if (editing && !refreshing) setWidgetValue(this, promptWidget, EDITED_PROMPT_PREFIX + textarea.value);
      });
      textarea.addEventListener("keydown", (event) => event.stopPropagation());
      textarea.addEventListener("keyup", (event) => event.stopPropagation());

      editButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (refreshing) return;
        setWidgetValue(this, refreshWidget, false);
        setEditing(!editing);
      });

      refreshButton.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!editing || refreshing) return;
        editing = false;
        textarea.value = "";
        setWidgetValue(this, promptWidget, "");
        setWidgetValue(this, refreshWidget, true);
        setRefreshing(true, "\u6b63\u5728\u5237\u65b0\u2026");
        this.graph?.setDirtyCanvas(true, true);
        try {
          const prompt = await buildPromptOnlyQueue(this.id);
          const queued = await api.queuePrompt(-1, prompt);
          this._ltx23RefreshPromptId = queued?.prompt_id || null;
        } catch (error) {
          console.error("[WenWu LTX-2.3] Prompt refresh failed", error);
          setWidgetValue(this, refreshWidget, false);
          setRefreshing(false, "\u5237\u65b0\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u63a7\u5236\u53f0");
        }
      });

      panel.append(topControls, textarea, refreshButton);
      const panelWidget = this.addDOMWidget("ltx23_prompt_panel", "div", panel, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 370,
      });
      if (panelWidget) panelWidget.computeSize = (width) => [width, 370];

      this._ltx23Panel = { textarea, status, editButton, refreshButton, promptWidget, refreshWidget, setEditing, setRefreshing };
      this.setSize([680, Math.max(this.size?.[1] || 0, 760)]);
      return result;
    };

    const originalConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigured?.apply(this, arguments);
      requestAnimationFrame(() => {
        if (!this._ltx23Panel) return;
        repairCoreWidgets(this);
        localizeInteractiveWidget(this.widgets?.find((widget) => widget.name === "duration_seconds"), "视频秒数");
        localizeInteractiveWidget(this.widgets?.find((widget) => widget.name === "frame_rate"), "视频帧率");
        localizeInteractiveWidget(this.widgets?.find((widget) => widget.name === "resolution"), "视频分辨率");
        this.setSize([680, Math.max(this.size?.[1] || 0, 760)]);
        this._ltx23Panel.textarea.value = displayedPrompt(this._ltx23Panel.promptWidget?.value);
        setWidgetValue(this, this._ltx23Panel.promptWidget, this._ltx23Panel.textarea.value);
        setWidgetValue(this, this._ltx23Panel.refreshWidget, false);
        this._ltx23Panel.setRefreshing(false);
        this._ltx23Panel.setEditing(false);
      });
      return result;
    };

    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = originalExecuted?.apply(this, arguments);
      const text = String(message?.optimized_prompt?.[0] || "");
      if (this._ltx23Panel && text) {
        this._ltx23Panel.textarea.value = text;
        setWidgetValue(this, this._ltx23Panel.promptWidget, text);
        setWidgetValue(this, this._ltx23Panel.refreshWidget, false);
        this._ltx23Panel.setRefreshing(false, "\u5237\u65b0\u5b8c\u6210");
        this._ltx23Panel.setEditing(false);
      }
      this.graph?.setDirtyCanvas(true, true);
      return result;
    };
  },
});
