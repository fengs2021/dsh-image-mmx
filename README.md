# dsh-image-mmx

给 DeepSeek Harness 的**文本模型**装一双眼睛：会话里粘贴/发送图片，自动调用 **mmx（MiniMax VLM）** 识别，识别结果注入模型上下文——DeepSeek 等纯文本模型也能理解图片内容。

## 能力

- **图片收发**：文本模型（`inputModalities` 无 image）也能发送含图消息，不再报「当前模型不支持图片」
- **自动识别**：图片落盘到会话工作区 `.attachments/`，自动执行 `mmx vision describe`（MiniMax VLM）
- **模型可见替换**：图片块在模型可见面替换为 `[图片N]:"<路径>"` + mmx 识别文本（画面元素、布局、报错/代码/界面文字逐字保留）
- **用户侧不变**：人类 transcript 照常渲染原图缩略图（可点击放大）
- **优雅降级**：mmx 未安装/未认证/超时（60s）/格式不支持（gif）时降级为路径文本
- **原生模型免打扰**：本身支持图片输入的模型走原生路径，不做任何处理

配套插件：[dsh-file-bridge](https://github.com/fengs2021/dsh-file-bridge)（📎 附件按钮上传 + send_files 文件下发 + explorer 跳转）。

## 依赖

本机已安装并认证 `mmx-cli`（MiniMax 官方 CLI）：

```bash
npm install -g mmx-cli
mmx auth login --api-key sk-xxxxx   # 或 OAuth
mmx vision describe --image test.png   # 验证可用
```

## 安装

前置：DeepSeek Harness `0.1.0-rc.6+`、Node.js 18+。

```bash
dsh plugin --profile web add 'github:fengs2021/dsh-image-mmx'
systemctl restart dsh-web
```

## 使用

1. 在 DSH Web 输入框**粘贴/拖入图片**（或配合 dsh-file-bridge 的 📎 附件按钮选图）
2. 发送后插件自动：落盘 → mmx 识别 → 模型直接读到识别结果并回复
3. 多图支持：每张图独立并行识别，按 `[图片1] [图片2]...` 编号

## 实现

基于 [dsh-image-bridge](https://github.com/haitang1/dsh-image-bridge)（MIT）机制改造：

1. 包装 `llm.resolveModelInfo` 为文本模型补 `image` 模态（放行 api-proxy 预检）
2. `agent/pre-step`：图片落盘 + 并行调用 `mmx vision describe`
3. `agent/request-error`：surface replace 把图片块换成识别文本，`retry` 重发

## License

MIT
