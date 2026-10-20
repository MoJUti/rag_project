# 知识库工作台前端

React＋TypeScript＋Vite，手写布局和样式，不依赖Streamlit。支持会话选择、新建、GLM/千问选择、流式回答、引用跳转与单次流程查看。文档管理、解析预览、切块配置及其他策略选择尚未接入，不展示假的可用入口。

## 启动

优先在项目根目录执行 `powershell -File scripts/start_web.ps1`，构建后用一个FastAPI服务提供页面与接口。默认地址为 `http://127.0.0.1:8765/`。

前端开发时可在本目录运行：

```powershell
npm ci
npm run dev
```

Vite将 `/api` 转发到本机8765端口，另开终端启动Python API：

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8765
```

后端默认GLM/low，千问可选；真实密钥只放根目录`.env`，不创建带密钥的Vite环境变量。浏览器只访问本站API，不直接访问模型供应商。

## 检查

```powershell
npm test
npm run build
npm run format
```

依赖锁在 `package-lock.json`；`node_modules`和`dist`不提交。默认Markdown不执行原始HTML，不从外部字体服务加载资源。SSE处理中文跨字节分包、错误终结事件和非正常断流；旧回答没有证据快照时，不把它的S1关联到另一回答。

测试通过不等于浏览器布局已由人工验收。详见 `docs/React工作台迁移与首轮验收.md`。
