import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "ShowAnything|WenWu",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.category !== "WenWu/Prompt") {
            return;
        }

        if (nodeData.name === "WenWuShowAndSaveAnything") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (!this.textWidget) {
                    this.textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", { multiline: true }], app).widget;
                    this.textWidget.inputEl.readOnly = true;
                    this.textWidget.inputEl.style.border = "none";
                    this.textWidget.inputEl.style.backgroundColor = "transparent";
                }

                const raw = message?.text ?? message?.ui?.text ?? "";
                const text = Array.isArray(raw) ? (raw[0] ?? "") : raw;
                this.textWidget.value = String(text);
                this.textWidget.inputEl.value = this.textWidget.value;
            };
        }
    },
});
