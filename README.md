# 扫描件自动裁剪（Project 17 · Auto Cropping of Image Scans）

一个本地运行的小工具：把扫描/拍摄的照片、明信片扫描成一张大图后，自动找出其中的照片区域，做**矩形裁剪 + 透视矫正**，再批量导出；自动检测不准时可以手动拖四角兜底。

技术栈：`HTML + CSS + 原生 JS` 前端 / `FastAPI` 接口层 / `OpenCV + NumPy + Pillow` 图像处理。

> 说明：本项目做的是**矩形区域裁剪**，不是抠图。输出保留矩形边界，不做透明背景抠像。

---

## 1. 功能清单

| 功能 | 状态 | 实现位置 |
| --- | --- | --- |
| 多张图片导入（支持拖放） | 必做 | `app/static/app.js` `uploadFiles()` / `app/main.py` `POST /api/upload` |
| 自动检测重点区域 | 必做 | `app/detector.py` `detect()` |
| 自动矩形裁剪 + 透视矫正 | 必做 | `app/detector.py` `crop_region()` |
| 检测失败的明确提示 | 必做 | `detect()` 返回 `reason`，前端状态条显示 |
| 手动兜底编辑（拖四角 / 整体平移 / 旋转 90° / 多候选切换） | 加分 | `app/static/app.js` 画布拖拽逻辑 |
| 结果对照与浏览 | 加分 | `app/static/app.js` `renderResults()` |
| 批量导出 | 加分 | `app/main.py` `POST /api/export` |
| 降采样（可选长边上限，只缩不放） | 文档要求 | `app/exporter.py` `downsample()` |
| 水印（文字、九宫格位置、透明度、字号比例） | 文档要求 | `app/exporter.py` `add_watermark()` |
| 自定义标签 + 文件名模板 | 文档要求 | `app/exporter.py` `build_stem()` |
| 导出附带 sidecar JSON（记录来源与四角坐标） | 加分 | `app/exporter.py` `export_item()` |
| 检测精度量化评估（IoU） | 加分 | `tools/eval.py` + `sample_data/` |
| 端到端冒烟测试 | 加分 | `tools/smoke_test.py` |

---

## 2. 目录结构

```
专题项目/
├── config.json              # 所有可调参数（检测阈值、上传限制、导出默认值）
├── requirements.txt
├── start.bat                # 一键启动（装依赖 + 开服务 + 开浏览器）
├── README.md
├── app/
│   ├── config.py            # 读取 config.json（缺字段时回落到默认值）
│   ├── storage.py           # 上传记录持久化（data/manifest.json）
│   ├── detector.py          # 检测 + 透视裁剪（核心算法）
│   ├── exporter.py          # 降采样 / 水印 / 命名 / 写盘
│   ├── main.py              # FastAPI 路由与静态资源挂载
│   └── static/              # index.html / style.css / app.js
├── tools/
│   ├── make_demo_data.py    # 生成合成扫描样本 + 真值
│   ├── eval.py              # 用 IoU 评估检测精度，输出 eval_report.csv
│   └── smoke_test.py        # 走 HTTP 层的端到端自测
├── sample_data/
│   ├── images/              # 6 张合成样本
│   └── labels/              # 5 份真值（scan_03 为双区域样本，无真值）
└── data/                    # 运行时生成：uploads / previews / crops / exports / manifest.json
```

---

## 3. 快速开始

### 方式一：双击 `start.bat`

会依次：切 UTF-8 代码页 → 安装依赖 → 延时 3 秒后打开 `http://127.0.0.1:8000` → 启动服务。

### 方式二：手动

```bash
cd 专题项目
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

然后浏览器打开 <http://127.0.0.1:8000>。

环境要求：Python 3.10+（开发验证于 3.13）、Windows / macOS / Linux 均可。首次安装 `opencv-python` 约需几十 MB。

---

## 4. 使用流程

1. 点「导入图片」或直接把文件拖进窗口（支持 jpg/jpeg/png/webp/bmp，单张 ≤ 60MB，一次 ≤ 30 张）。
2. 每张图自动检测，左侧列表显示状态：绿色=检测成功，橙色=置信度偏低/未检出，灰色=已丢弃。
3. 中间画布显示检测到的四边形（青色框 + 四角手柄）：
   - 拖动角点微调；在框内拖动可整体平移；
   - 多区域图片可在右上角候选条里切换；
   - 点「旋转 90°」修正方向。
4. 点「接受」生成裁剪成品；右侧「成品」栏显示结果，可全屏对照原图。
5. 底部设置导出参数：格式（jpg/png）、质量、降采样长边、水印文字/位置/透明度、标签、文件名模板、输出目录。
6. 点「导出全部已接受」批量写出文件（默认写到 `data/exports/<时间戳>/`）。

文件名模板支持四个占位符：`{name}` 原文件名、`{index}` 序号、`{label}` 自定义标签、`{date}` 导出日期（如 `20261102`）。

---

## 5. 算法说明

检测在缩放到长边 ≤ `detect_max_side`（默认 1600）的灰度图上进行，坐标最后按比例还原到原图，裁剪始终在**原分辨率**上做。

三条候选策略并行，合并后按包围盒重合度去重（IoU > 0.6 只留置信度高的）：

| 策略 | 做法 | 适用 |
| --- | --- | --- |
| A 底色差 | 取画面四周 5% 边框带的中位色作背景估计，逐像素算最大通道差，阈值取 `max(8, median + 2.5 × 1.4826 × MAD)`（自适应噪声），再闭/开运算、找外轮廓 | 卡片与背景有色差 |
| B Canny 边缘 | 以灰度中位数自适应上下阈值提边，膨胀 + 闭运算把边连成环，找外轮廓 | 白底白卡等低对比场景 |
| C 梯度兜底 | A、B 都没结果时，Sobel 幅值 > `max(18, p90)` 取强梯度，重闭运算成环 | 背景估计完全失效 |

轮廓 → 四边形的规则：

- 面积占比需在 `[min_area_ratio, max_area_ratio]` = `[1.5%, 92%]` 内；
- 凸包实心度（轮廓面积 / 凸包面积）≥ `min_solidity` = 0.85，排除树状/镂空轮廓；
- `cv2.approxPolyDP` 从 eps=0.015 递增到 0.16 找恰好 4 点的凸四边形；找不到则退化为 `minAreaRect`；
- 置信度 = `0.45 × 填充度 + 0.55 × 四边梯度支持度`（沿四条边采样 Sobel 幅值，除以 p99 归一化）；
- 低于 `min_confidence` = 0.45 时仍返回候选，但标记 `low_confidence` 让前端提示。

裁剪：`order_quad` 用 sum/diff 把四角排成 TL→TR→BR→BL，`getPerspectiveTransform` + `warpPerspective` 输出正向矩形；输出宽高取对边长度的较大值；额外向内收缩 `inset_px`（默认 2px）避免带进背景残留。

读图用 Pillow（`ImageOps.exif_transpose` 修正手机拍摄方向），写图用 `cv2.imencode` + `tofile`，避开 OpenCV 在 Windows 下读不了中文路径的问题。

---

## 6. config.json 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `detector.detect_max_side` | 1600 | 检测时缩放的最长边；越大越准、越慢 |
| `detector.border_ratio` | 0.05 | 估计背景色时取四周多宽的边框带（占短边比例） |
| `detector.min_area_ratio` | 0.015 | 候选区域最小面积占比 |
| `detector.max_area_ratio` | 0.92 | 候选区域最大面积占比（防止把整张图当目标） |
| `detector.min_solidity` | 0.85 | 最小实心度，滤掉不规则轮廓 |
| `detector.close_kernel` | 9 | 策略 A 的闭运算核大小 |
| `detector.max_regions` | 4 | 单张图最多返回几个候选区域 |
| `detector.min_confidence` | 0.45 | 低于此值判为「置信度不足」 |
| `detector.inset_px` | 2 | 裁剪时四边向内收缩的像素数 |
| `upload.max_files` | 30 | 单次上传张数上限 |
| `upload.max_file_mb` | 60 | 单张文件大小上限 |
| `upload.preview_max_side` | 1600 | 前端预览图最长边 |
| `upload.allowed_ext` | jpg/jpeg/png/webp/bmp | 允许的扩展名 |
| `export.default_format` | jpg | 默认导出格式 |
| `export.default_quality` | 92 | 默认 JPEG 质量 |
| `export.write_sidecar` | true | 默认是否写同名 .json |

改完 `config.json` 需重启服务生效。

---

## 7. API 接口

| 方法 + 路径 | 请求 | 返回 |
| --- | --- | --- |
| `GET /api/health` | — | `{ok, time}` |
| `GET /api/config` | — | 上传限制与导出默认值（前端初始化用） |
| `POST /api/upload` | multipart，字段名 `files`（可多个） | `{items: [{id, name, url, width, height}]}`，单个失败项含 `error` |
| `POST /api/detect` | `{image_id}` | `{status: ok/failed, reason: null/no_region/low_confidence, regions: [{quad, confidence, area_ratio}], width, height, detect_ms}` |
| `POST /api/crop` | `{image_id, quad: [[x,y]×4], rotate: 0/90/180/270}` | `{url, width, height}` |
| `POST /api/export` | `{items: [{image_id, quad, rotate}], options: {...}}` | `{output_dir, files: [{name, path, width, height}], errors, count}` |

`quad` 使用**原图像素坐标**，按 TL→TR→BR→BL 顺序最稳（服务端会做顺序整理与边界裁剪）。`options` 字段见 `app/main.py` 的 `ExportOptions`。

静态资源：`/media/*` 指向 `data/`（预览图、裁剪结果），`/` 指向前端页面。

---

## 8. 评估方法与本项目结果

`tools/eval.py` 的评估方式：

- 真值格式：与图片同名的 `.json`，内容为 `{"quad": [[x,y]×4]}`（原图像素坐标）；
- 预测取置信度/面积最大的候选；
- 把预测与真值多边形缩放到长边 ≤ 1200 的掩膜上，用 `cv2.fillPoly` 计算**多边形 IoU**；
- 判定：IoU ≥ 0.85 记为成功，0.70–0.85 记为部分成功，其余失败。

本项目在 `sample_data/`（**6 张合成扫描图，5 张有真值**）上的结果：

| 图片 | IoU | 状态 |
| --- | --- | --- |
| scan_01_white_card | 0.9954 | ok |
| scan_02_postcard_dark | 0.9775 | ok |
| scan_04_white_on_white | 0.9834 | ok |
| scan_05_perspective | 0.9979 | ok |
| scan_06_small_print | 0.9854 | ok |
| **汇总** | **平均 0.9879** | **ok=5, partial=0, fail=0** |

`scan_03_two_photos` 是双区域样本（检测正确返回 2 个区域），没有真值所以不计入 IoU 统计。

复现：

```bash
python tools/make_demo_data.py    # 重新生成合成样本与真值
python tools/eval.py              # 打印表格并写 eval_report.csv
python tools/smoke_test.py        # 端到端 HTTP 自测，应 11/11 PASS
```

> **重要声明**：上述数字来自**合成样本**，背景/卡片用渐变与噪声程序生成，不能代表真实扫描件的难度。要在报告里给出可信结论，请用同一格式自建真实标注集：把真实扫描图放入 `sample_data/images/`（或另建目录），用任意标注工具（LabelMe、CVAT，或本工具手动拖角后把 quad 存成 json）产出同名 `.json`，再改 `tools/eval.py` 的输入目录运行。真实数据上预计会出现白底白卡、卡片叠压、非矩形目标等失败情况。

---

## 9. 已知限制

- **不支持** TIFF / HEIC / RAW；单张上限 60MB。
- 白底白卡、极低对比度、强反光时可能漏检或框偏，需手动拖角兜底。
- 多张卡片互相叠压时，可能只识别到最外层轮廓。
- 非矩形目标（圆盘、不规则照片）会被退化成最小外接旋转矩形。
- 卡片本身有粗白边时，边界可能落在白边内侧或外侧，取决于对比度；可在接受前手动外扩。
- 不做内容级修复（去网纹、去灰尘、自动色彩校正）。
- 服务是**单机本地工具**：无用户系统、无并发控制，`manifest.json` 为简单 JSON 持久化，不适合多用户同时使用。
- 输出目录必须是绝对路径（安全校验，防止意外覆盖）。

---

## 10. 后续可扩展方向

1. 接入 OCR/内容识别，自动判断卡片类型并打标签。
2. 用真实标注集训练一个轻量分割模型（如 U-Net / MobileNet 分割头）替换纯几何检测。
3. 批量任务队列 + 进度条 + 断点续跑（当前是同步逐张处理）。
4. 拖入文件夹、按子目录保持输出结构。
5. 导出 PDF 拼版 / 联系表（contact sheet）。
6. 打包成单文件桌面应用（PyInstaller + 本地端口自启）。

---

## 11. 交付/交接要点

- 所有可调参数集中在 `config.json`，不需要改代码就能适配不同扫描仪背景。
- 算法入口是 `app/detector.py::detect`，输入 BGR 数组 + 配置字典，输出 `(regions, reason)`，与 Web 层解耦，可单独调用/替换。
- 接口契约在第 7 节；前端只通过这四个接口与后端交互，改前端不影响算法。
- 运行时数据全部落在 `data/`，删除该目录即可清空所有上传与导出记录（`exports/` 下若有手动指定目录的文件不会被删）。
- 验证命令：`tools/eval.py`（精度）与 `tools/smoke_test.py`（链路），改动算法后两者都应重跑。
